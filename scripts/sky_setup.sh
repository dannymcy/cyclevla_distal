#!/bin/bash
set -euo pipefail

sudo apt-get update && sudo apt-get install -y \
  cmake \
  ffmpeg \
  g++ \
  libegl1 \
  libexpat1 \
  libfontconfig1-dev \
  libgl1 \
  libglvnd0 \
  libmagickwand-dev \
  libopengl0 \
  unzip

curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
cd ~/sky_workdir

pixi install

bash ~/sky_workdir/scripts/setup_libero.sh

cd ~/sky_workdir
pixi run python ~/sky_workdir/.pixi/envs/default/lib/python3.12/site-packages/robosuite/scripts/setup_macros.py
