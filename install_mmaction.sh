#!/usr/bin/env bash
set -e  # stop on first error
set -o pipefail

echo "Installing MMAction2 dependencies via mim..."
for pkg in mmcls==0.25.0 mmengine==0.10.4 mmcv==2.1.0 mmpretrain==1.2.0; do
    echo "Installing $pkg..."
    python -m mim install "$pkg"
done

TAG="v1.2.0"
TMPDIR=$(mktemp -d)
REPO="$TMPDIR/mmaction2"

echo "Cloning MMACTION2 $TAG..."
git clone --branch "$TAG" --single-branch --depth 1 https://github.com/open-mmlab/mmaction2.git "$REPO"

# Patch missing __init__.py
DRN_DIR="$REPO/mmaction/models/localizers/drn"
if [ ! -f "$DRN_DIR/__init__.py" ]; then
    echo "Patching missing __init__.py in $DRN_DIR"
    touch "$DRN_DIR/__init__.py"
fi

echo "Installing MMACTION2 from local clone..."
python -m pip install "$REPO"

echo "MMAction2 installation complete!"