#!/bin/bash
# One-shot libero / libero_plus bootstrap for a new machine.
#
# - Locates the installed `libero` (hf-libero) and `libero_plus` packages
#   under $VENV_DIR (default: $PWD/.pixi/envs/default) — `find -L` because
#   the env may be a symlink (e.g. Isambard scratch).
# - Writes ~/.libero/config.yaml and ~/.libero_plus/config.yaml so each
#   package's `__init__.py` skips its interactive `input()` prompt on
#   first import.
# - Downloads the LIBERO-plus assets bundle from HF into the libero_plus
#   site-packages dir if not already present, so AsyncVectorEnv workers
#   don't race to fetch them at first eval.
#
# Idempotent: safe to re-run.
set -euo pipefail

VENV_DIR="${VENV_DIR:-$PWD/.pixi/envs/default}"
ASSETS_URL="${ASSETS_URL:-https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip}"

LIBERO_DIR=$(find -L "$VENV_DIR" -type d -path "*/libero/libero" -not -path "*libero_plus*" -print -quit)
LIBERO_PLUS_DIR=$(find -L "$VENV_DIR" -type d -path "*/libero_plus/libero_plus" -print -quit)

if [ -z "$LIBERO_PLUS_DIR" ]; then
  echo "libero_plus install not found under $VENV_DIR — run 'pixi install' first" >&2
  exit 1
fi

mkdir -p "$HOME/.libero_plus"
cat > "$HOME/.libero_plus/config.yaml" <<EOF
assets: $LIBERO_PLUS_DIR/assets
bddl_files: $LIBERO_PLUS_DIR/bddl_files
benchmark_root: $LIBERO_PLUS_DIR
datasets: $LIBERO_PLUS_DIR/../datasets
init_states: $LIBERO_PLUS_DIR/init_files
EOF

if [ -n "$LIBERO_DIR" ]; then
  mkdir -p "$HOME/.libero"
  cat > "$HOME/.libero/config.yaml" <<EOF
assets: $LIBERO_DIR/assets
bddl_files: $LIBERO_DIR/bddl_files
benchmark_root: $LIBERO_DIR
datasets: $LIBERO_DIR/../datasets
init_states: $LIBERO_DIR/init_files
EOF
fi

LOCAL_BASE="${LOCAL_BASE:-/local/user/$UID}"
LOCAL_ASSETS="$LOCAL_BASE/libero_assets"
ASSETS_CACHE="${ASSETS_CACHE:-${SCRATCHDIR:-$HOME}/libero_assets.zip}"

if [ ! -d "$LOCAL_ASSETS" ]; then
  if [ ! -f "$ASSETS_CACHE" ]; then
    mkdir -p "$(dirname "$ASSETS_CACHE")"
    curl -L -o "$ASSETS_CACHE" "$ASSETS_URL"
  fi
  mkdir -p "$LOCAL_BASE"
  TMP_EXTRACT=$(mktemp -d -p "$LOCAL_BASE")
  TOTAL_BYTES=$(unzip -l "$ASSETS_CACHE" | tail -1 | awk '{print $1}')
  unzip -q "$ASSETS_CACHE" -d "$TMP_EXTRACT" &
  UNZIP_PID=$!
  START=$SECONDS
  PREV_BYTES=0
  PREV_TIME=0
  while kill -0 "$UNZIP_PID" 2>/dev/null; do
    sleep 10
    BYTES=$(du -sb "$TMP_EXTRACT" 2>/dev/null | awk '{print $1}')
    ELAPSED=$((SECONDS - START))
    DELTA_B=$((BYTES - PREV_BYTES))
    DELTA_T=$((ELAPSED - PREV_TIME))
    if [ "$DELTA_T" -gt 0 ] && [ "$DELTA_B" -gt 0 ] && [ "$TOTAL_BYTES" -gt 0 ]; then
      RATE=$((DELTA_B / DELTA_T))
      REMAIN=$((TOTAL_BYTES - BYTES))
      if [ "$REMAIN" -gt 0 ] && [ "$RATE" -gt 0 ]; then
        ETA_S=$((REMAIN / RATE))
        PCT=$((BYTES * 100 / TOTAL_BYTES))
        printf 'Unzipping libero plus assets: %s/%s (%d%%, ETA %s)\n' \
          "$(numfmt --to=iec "$BYTES")" \
          "$(numfmt --to=iec "$TOTAL_BYTES")" \
          "$PCT" "$(date -ud "@$ETA_S" +%H:%M:%S)"
      fi
    fi
    PREV_BYTES=$BYTES
    PREV_TIME=$ELAPSED
  done
  wait "$UNZIP_PID"
  mv "$(find "$TMP_EXTRACT" -type d -name assets -print -quit)" "$LOCAL_ASSETS"
  rm -rf "$TMP_EXTRACT"
fi

ln -sfn "$LOCAL_ASSETS" "$LIBERO_PLUS_DIR/assets"

echo "libero_plus ready at $LIBERO_PLUS_DIR"
[ -n "$LIBERO_DIR" ] && echo "libero ready at $LIBERO_DIR"
