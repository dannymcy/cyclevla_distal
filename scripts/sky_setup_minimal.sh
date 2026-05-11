#!/bin/bash
set -euo pipefail

sudo apt-get update && sudo apt-get install -y ffmpeg

curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
cd ~/sky_workdir

pixi install
