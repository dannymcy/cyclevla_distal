"""Action-variance-based per-step rewards for value training.

For each frame in a dataset, draw ``num_samples`` action chunks from the base
policy with the same observation but different denoising noise, and use the
across-sample variance (averaged over chunk timestep and action dim) as the
per-frame uncertainty signal. Plugged into the same
``normalize_distances_to_rewards`` pipeline as ``rewards/maha.py`` and
``rewards/knn.py`` so high variance maps to low reward.

Prefix reuse: the SmolVLM/PaliGemma prefix (vision + language → KV cache) is
identical across the ``num_samples`` branches for a given frame, so we run it
once per frame and then expand the KV cache along the batch dim for the
flow-matching denoising loop. This avoids ``num_samples``× redundant prefix
work, which dominates the per-frame cost.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, make_att_2d_masks
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from torch.utils.data import DataLoader, Subset


def infer_batch_size(batch: dict) -> int:
    for v in batch.values():
        if isinstance(v, torch.Tensor):
            return int(v.shape[0])
    raise ValueError("Batch contains no tensors; cannot infer batch size.")


def repeat_past_key_values(past_key_values, num_samples: int):
    """Repeat KV cache along the batch dim. Handles transformers ``Cache`` + tuples."""
    if hasattr(past_key_values, "batch_repeat_interleave"):
        past_key_values.batch_repeat_interleave(num_samples)
        return past_key_values
    return tuple(
        (
            k.repeat_interleave(num_samples, dim=0),
            v.repeat_interleave(num_samples, dim=0),
        )
        for (k, v) in past_key_values
    )


def sample_action_chunks_with_shared_prefix(
    policy: PI05Policy,
    batch: dict,
    *,
    num_samples: int,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Sample ``num_samples`` action chunks per frame, sharing the VLM prefix.

    ``batch`` is a size-k preprocessed batch. ``noise`` has shape
    ``(k * num_samples, chunk_size, max_action_dim)``. Returns actions of shape
    ``(k * num_samples, chunk_size, max_action_dim)`` (unpadded by the caller).
    """
    model = policy.model
    config = policy.config
    assert isinstance(config, PI05Config)

    images, img_masks = policy._preprocess_images(batch)  # noqa: SLF001
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Build prefix on the size-k batch.
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = (torch.cumsum(prefix_pad_masks, dim=1) - 1).long()
    prefix_att_2d_4d = model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001
    lang_cfg = model.paligemma_with_expert.paligemma.model.language_model.config
    lang_cfg._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_4d,
        position_ids=prefix_pos,  # ty: ignore[invalid-argument-type]
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],  # ty: ignore[invalid-argument-type]
        use_cache=True,
    )

    # Expand prefix state to (k * num_samples, ...) for the denoising loop.
    prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_samples, dim=0)
    past_key_values = repeat_past_key_values(past_key_values, num_samples)

    bsize = noise.shape[0]
    num_steps = config.num_inference_steps
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.tensor(t, dtype=torch.float32, device=noise.device).expand(
            bsize
        )
        v_t = model.denoise_step(prefix_pad_masks, past_key_values, x_t, t_tensor)
        x_t = x_t + dt * v_t

    return x_t


@torch.no_grad()
def compute_action_variance_for_dataset(
    dataset: LeRobotDataset,
    policy_path: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    num_samples: int,
    noise_seed: int,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    """Return per-frame action-variance scalars.

    For each frame, samples ``num_samples`` action chunks with independent
    noise, then averages the across-sample variance over the action chunk
    (timesteps and action dims). Output is a length-N float64 array aligned
    with ``frame_indices`` (or the full dataset if None).
    """
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = Path(policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, PI05Policy)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    loader_ds: LeRobotDataset | Subset = (
        Subset(dataset, frame_indices) if frame_indices is not None else dataset
    )
    loader = DataLoader(
        loader_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    config = policy.config
    assert isinstance(config, PI05Config)
    assert config.output_features is not None
    chunk_size = config.chunk_size
    max_action_dim = config.max_action_dim
    action_dim = config.output_features[ACTION].shape[0]
    generator = torch.Generator(device=device).manual_seed(noise_seed)

    all_vars: list[float] = []
    total = len(loader)
    logging.info(
        f"Action-variance: {total} batches "
        f"(batch_size={batch_size}, num_samples={num_samples}, "
        f"chunk_size={chunk_size}, action_dim={action_dim})"
    )
    start = time.monotonic()
    try:
        for i, batch in enumerate(loader):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch = preprocessor(batch)
            k = infer_batch_size(batch)
            noise = torch.randn(
                (k * num_samples, chunk_size, max_action_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            actions = sample_action_chunks_with_shared_prefix(
                policy, batch, num_samples=num_samples, noise=noise
            )
            actions = actions[:, :, :action_dim]
            actions = actions.view(k, num_samples, chunk_size, action_dim)
            # Variance across the N samples, then mean over timestep & action dim.
            per_frame = actions.var(dim=1, unbiased=False).mean(dim=(1, 2))
            all_vars.extend(per_frame.detach().float().cpu().tolist())

            done = i + 1
            if done % 50 == 0 or done == total:
                elapsed = time.monotonic() - start
                eta = elapsed * (total - done) / done
                logging.info(
                    f"Action-variance: {done}/{total} "
                    f"[elapsed {elapsed:.0f}s, eta {eta:.0f}s]"
                )
    finally:
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(all_vars, dtype=np.float64)
