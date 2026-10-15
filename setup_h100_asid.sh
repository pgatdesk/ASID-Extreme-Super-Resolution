#!/usr/bin/env bash
# Recreates the ASID x8 Colab workflow on the allocated Linux/H100 container.
# Run: bash setup_h100_asid.sh [workspace] [checkpoint-directory]
set -euo pipefail

WORKSPACE="${1:-$HOME/asid_h100}"
CHECKPOINTS="${2:-$WORKSPACE/checkpoints}"
REPO="$WORKSPACE/ASID-Extreme-Super-Resolution"
DATA="$WORKSPACE/DIV2K"

mkdir -p "$WORKSPACE" "$CHECKPOINTS"
cd "$WORKSPACE"

if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/MayankV004/ASID-Extreme-Super-Resolution.git "$REPO"
fi

python3 -m venv "$WORKSPACE/.venv"
source "$WORKSPACE/.venv/bin/activate"
python -m pip install --upgrade pip

# Install the CUDA-enabled PyTorch build recommended by the container's driver,
# then the project's pinned dependencies.
pip install torch torchvision
pip install -r "$REPO/requirements.txt"
pip install thop tensorboardX einops timm opencv-python pyyaml tqdm matplotlib

mkdir -p "$DATA"
cd "$DATA"
if [ ! -f DIV2K_train_HR.zip ]; then
  wget -c https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
fi
if [ ! -d DIV2K_train_HR ]; then
  unzip -q DIV2K_train_HR.zip
fi

# Generate exactly the bicubic x8 LR images expected by the notebook.
DATA_DIR="$DATA" python - <<'PY'
import glob, os
from PIL import Image
from tqdm import tqdm

root = os.environ['DATA_DIR']
source = os.path.join(root, 'DIV2K_train_HR')
target = os.path.join(root, 'DIV2K_train_LR_bicubic', 'X8')
os.makedirs(target, exist_ok=True)
for image_path in tqdm(sorted(glob.glob(os.path.join(source, '*.png')))):
    name = os.path.basename(image_path).replace('.png', 'x8.png')
    output = os.path.join(target, name)
    if not os.path.exists(output):
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            width, height = image.size
            width, height = (width // 8) * 8, (height // 8) * 8
            image.crop((0, 0, width, height)).resize(
                (width // 8, height // 8), Image.Resampling.BICUBIC
            ).save(output, 'PNG')
PY

cd "$REPO"
DATA_DIR="$DATA" python - <<'PY'
import json, os
with open('env/env.json') as f:
    config = json.load(f)
config['path']['dataset_paths']['DIV2K'] = os.environ['DATA_DIR']
with open('env/env.json', 'w') as f:
    json.dump(config, f, indent=2)
PY

# Keep both experiments' checkpoints outside the repository, like the Google
# Drive link in Colab. Each run gets an independent persistent directory.
link_checkpoints() {
  local run="$1"
  local destination="$CHECKPOINTS/$run"
  mkdir -p "$destination" "train_logs/$run"
  if [ -e "train_logs/$run/checkpoints" ] && [ ! -L "train_logs/$run/checkpoints" ]; then
    mv "train_logs/$run/checkpoints" "train_logs/$run/checkpoints.bak.$(date +%s)"
  fi
  ln -sfn "$destination" "train_logs/$run/checkpoints"
}
link_checkpoints 'ASID_PreUpsample_X8_DIV2K'
link_checkpoints 'ASID_X8_DIV2K'

nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable in this virtual environment.'
print('Using:', torch.cuda.get_device_name(0))
PY

echo
echo 'Setup complete. Start training with:'
echo "  source '$WORKSPACE/.venv/bin/activate'"
echo "  cd '$REPO'"
echo '  python train.py -v ASID_PreUpsample_X8_DIV2K -p train --train_yaml train_ASID_PreUpsample_X8_DIV2K.yaml'
echo '  python train.py -v ASID_X8_DIV2K -p train --train_yaml train_ASID_X8_DIV2K.yaml'
