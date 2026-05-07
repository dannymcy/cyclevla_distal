"""Evaluate Mahalanobis distance as a failure predictor via AUROC.

Loads pre-computed Mahalanobis stats, embeds dataset frames, computes
per-frame distances, aggregates per episode (mean), and reports AUROC
against episode success labels.
"""

import json
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import draccus
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from safetensors.numpy import load_file
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from distal.collect_libero_plus import sample_task_ids
from distal.compute_maha_stats import compute_mahalanobis_np, embed_siglip_pooled

PERTURBATION_PATTERNS = {
    "language": re.compile(r"_language_"),
    "view": re.compile(r"_view_"),
    "light": re.compile(r"_light_"),
    "table": re.compile(r"_(?:table|tb)_\d+"),
    "add": re.compile(r"_add_\d+"),
    "level": re.compile(r"_(?:moved_)?level\d+_sample\d+"),
    "noise": re.compile(r"_noise_\d+"),
}


def perturbation_kinds(variant_name: str) -> set[str]:
    return {k for k, p in PERTURBATION_PATTERNS.items() if p.search(variant_name)}


def replay_variant_names(
    suites: list[str], per_cell: int, seed: int, max_tasks: int | None
) -> list[str]:
    """Reconstruct per-episode variant names from a collect_libero_plus run.

    Mirrors the iteration order of ``distal.collect_libero_plus.main``:
    ``for suite in suites: for tid in sample_task_ids(suite)[:max_tasks]``.
    Independent of ``parallel_envs`` since chunking preserves order.

    Note on the off-by-one: ``task_classification.json`` ids are 1-indexed
    (1..N) but ``LiberoEnv`` indexes ``suite.tasks[task_id]`` zero-indexed,
    so the variant actually rolled out for ``tid=K`` is ``entries[K]`` (i.e.
    JSON id=K+1). For ``tid=N`` (max) ``suite.tasks[N]`` is out of range and
    ``LiberoEnv`` would have crashed at collection time — those tids are
    skipped here.
    """
    classif = json.loads(
        (files("libero.libero") / "benchmark" / "task_classification.json").read_text()
    )
    names: list[str] = []
    for suite_name in suites:
        entries = classif[suite_name]
        ids = sample_task_ids(suite_name, per_cell=per_cell, seed=seed)
        if max_tasks is not None:
            ids = ids[:max_tasks]
        skipped = 0
        for tid in ids:
            if tid >= len(entries):
                skipped += 1
                continue
            names.append(entries[tid]["name"])
        if skipped:
            print(
                f"[replay] {suite_name}: skipped {skipped} tid(s) >= "
                f"{len(entries)} (would have crashed LiberoEnv at collect time)"
            )
    return names


@dataclass
class MahaAurocConfig:
    policy_path: str = "lerobot/pi05-libero"
    dataset_repo_id: str = "reece-omahoney/pi05-libero-plus"
    maha_stats_repo_id: str = "reece-omahoney/pi05-libero-plus-maha-stats-siglip"
    episodes_per_kind: int = 30
    min_per_class: int = 10
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

    # Collection config used to produce dataset_repo_id. Must match the values
    # passed to distal.collect_libero_plus, otherwise the variant replay will
    # not align with dataset episode_index.
    suites: list[str] = field(
        default_factory=lambda: [
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ]
    )
    per_cell: int = 1
    collect_seed: int = 0
    max_tasks: int | None = None

    # Set False for base LIBERO (no perturbations); skips variant replay and
    # per-kind AUROC, falling back to a balanced sample over all episodes.
    per_kind: bool = True


@draccus.wrap()
def main(cfg: MahaAurocConfig):
    init_logging()
    register_third_party_plugins()

    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load maha stats
    stats_path = hf_hub_download(
        repo_id=cfg.maha_stats_repo_id,
        filename="stats.safetensors",
        repo_type="dataset",
        force_download=True,
    )
    stats = load_file(stats_path)
    gauss_mean = stats["mean"]
    gauss_cov_inv = stats["cov_inv"]
    print(
        f"Loaded Mahalanobis stats: mean {gauss_mean.shape}, "
        f"cov_inv {gauss_cov_inv.shape}"
    )

    # Load dataset
    dataset = LeRobotDataset(repo_id=cfg.dataset_repo_id, vcodec="auto")
    episode_index = np.array(dataset.hf_dataset["episode_index"])
    success = np.array(dataset.hf_dataset["success"])

    unique_episodes = np.unique(episode_index)
    ep_success_map = {
        int(ep): bool(success[episode_index == ep][0]) for ep in unique_episodes
    }
    rng = np.random.default_rng(cfg.seed)
    selected_episodes: set[int] = set()

    if cfg.per_kind:
        # Replay collection order to map episode_index -> variant name.
        variant_names = replay_variant_names(
            cfg.suites, cfg.per_cell, cfg.collect_seed, cfg.max_tasks
        )
        if len(variant_names) != len(unique_episodes):
            print(
                f"[replay] WARNING: replay produced {len(variant_names)} variants "
                f"but dataset has {len(unique_episodes)} episodes. Per-kind AUROC "
                f"alignment is unreliable — investigate before trusting results."
            )
        ep_to_variant = {
            int(ep): name for ep, name in zip(unique_episodes, variant_names)
        }

        # Per-kind balanced subsetting: for each perturbation kind, sample up
        # to episodes_per_kind episodes, half success / half failure. Episodes
        # can appear in multiple kind buckets (kinds stack), so the unique-
        # episode union is smaller than 7 * episodes_per_kind.
        print(
            f"Per-kind balanced sampling, target {cfg.episodes_per_kind} "
            f"per kind ({cfg.episodes_per_kind // 2} succ + "
            f"{cfg.episodes_per_kind - cfg.episodes_per_kind // 2} fail):"
        )
        print(f"  {'kind':<10}  {'succ':>5}  {'fail':>5}  {'note':<30}")
        for kind in PERTURBATION_PATTERNS:
            succ_pool = np.array(
                [
                    ep
                    for ep in ep_to_variant
                    if ep_success_map[ep]
                    and kind in perturbation_kinds(ep_to_variant[ep])
                ],
                dtype=int,
            )
            fail_pool = np.array(
                [
                    ep
                    for ep in ep_to_variant
                    if not ep_success_map[ep]
                    and kind in perturbation_kinds(ep_to_variant[ep])
                ],
                dtype=int,
            )
            rng.shuffle(succ_pool)
            rng.shuffle(fail_pool)

            target = cfg.episodes_per_kind
            n_succ = min(target // 2, len(succ_pool))
            n_fail = min(target - n_succ, len(fail_pool))
            n_succ = min(target - n_fail, len(succ_pool))
            note = ""
            if min(n_succ, n_fail) < cfg.min_per_class:
                note = f"BELOW min_per_class={cfg.min_per_class}"
            print(f"  {kind:<10}  {n_succ:>5}  {n_fail:>5}  {note:<30}")

            for ep in succ_pool[:n_succ].tolist() + fail_pool[:n_fail].tolist():
                selected_episodes.add(int(ep))

        print(
            f"\nUnion of per-kind selections: {len(selected_episodes)} unique "
            f"episodes (out of {len(unique_episodes)} total)"
        )
    else:
        # Base LIBERO: balanced sample over all episodes, no per-kind logic.
        ep_to_variant = {}
        succ_pool = np.array(
            [ep for ep in unique_episodes if ep_success_map[int(ep)]], dtype=int
        )
        fail_pool = np.array(
            [ep for ep in unique_episodes if not ep_success_map[int(ep)]], dtype=int
        )
        rng.shuffle(succ_pool)
        rng.shuffle(fail_pool)
        target = cfg.episodes_per_kind
        n_succ = min(target // 2, len(succ_pool))
        n_fail = min(target - n_succ, len(fail_pool))
        n_succ = min(target - n_fail, len(succ_pool))
        print(
            f"Balanced sampling (no perturbation kinds): "
            f"{n_succ} succ + {n_fail} fail "
            f"(out of {len(unique_episodes)} total)"
        )
        for ep in succ_pool[:n_succ].tolist() + fail_pool[:n_fail].tolist():
            selected_episodes.add(int(ep))

    # Get frame indices for selected episodes
    frame_mask = np.isin(episode_index, list(selected_episodes))
    frame_indices = np.where(frame_mask)[0]

    # Load policy
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = Path(cfg.policy_path)
    policy_cfg.device = str(device)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    assert isinstance(policy, PI05Policy)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(policy_cfg.pretrained_path)
    )

    # Embed and compute distances
    subset = Subset(dataset, frame_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_dists: list[float] = []
    for batch in tqdm(loader, desc="Computing Maha distances"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = preprocessor(batch)
        with torch.no_grad():
            emb = embed_siglip_pooled(policy, batch)
            dists = compute_mahalanobis_np(emb.cpu().numpy(), gauss_mean, gauss_cov_inv)
        all_dists.extend(dists.tolist())

    distances = np.array(all_dists)
    selected_episode_index = episode_index[frame_indices]
    selected_success = success[frame_indices]

    # Aggregate per episode: mean distance and success label
    ep_mean_dist = {}
    ep_success = {}
    for ep in selected_episodes:
        mask = selected_episode_index == ep
        ep_mean_dist[ep] = distances[mask].mean()
        ep_success[ep] = bool(selected_success[mask][0])

    episodes = sorted(selected_episodes)
    scores = np.array([ep_mean_dist[ep] for ep in episodes])
    labels = np.array([not ep_success[ep] for ep in episodes])  # failure = positive

    n_fail = labels.sum()
    n_success = len(labels) - n_fail
    print(f"\nEpisodes: {len(labels)} ({n_success} success, {n_fail} failure)")

    if n_fail == 0 or n_success == 0:
        print("Cannot compute AUROC: only one class present.")
        return
    print(f"AUROC (mean Maha → failure): {roc_auc_score(labels, scores):.4f}")

    if not cfg.per_kind:
        return

    kinds_per_ep = [
        perturbation_kinds(ep_to_variant[ep]) if ep in ep_to_variant else None
        for ep in episodes
    ]
    n_unlabeled = sum(1 for k in kinds_per_ep if k is None)
    if n_unlabeled:
        print(
            f"[replay] {n_unlabeled}/{len(episodes)} selected episodes had no "
            f"replayed variant; excluded from per-kind AUROC."
        )

    print("\nPer-kind AUROC (episodes containing each kind):")
    print(f"  {'kind':<10}  {'n':>4}  {'succ':>5}  {'fail':>5}  {'auroc':>8}")
    for kind in PERTURBATION_PATTERNS:
        mask = np.array([ks is not None and kind in ks for ks in kinds_per_ep])
        if mask.sum() == 0:
            continue
        sub_labels = labels[mask]
        sub_scores = scores[mask]
        n_f = int(sub_labels.sum())
        n_s = int(len(sub_labels) - n_f)
        if n_f == 0 or n_s == 0:
            print(
                f"  {kind:<10}  {len(sub_labels):>4}  {n_s:>5}  {n_f:>5}  "
                f"{'-':>8} (single class)"
            )
            continue
        a = roc_auc_score(sub_labels, sub_scores)
        print(f"  {kind:<10}  {len(sub_labels):>4}  {n_s:>5}  {n_f:>5}  {a:>8.4f}")


if __name__ == "__main__":
    main()
