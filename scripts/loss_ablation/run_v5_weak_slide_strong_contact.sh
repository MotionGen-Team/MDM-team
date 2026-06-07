#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m train.train_mdm \
  --save_dir checkpoints/loss_v5_weak_slide_strong_contact \
  --arch plan_one \
  --lambda_hml_contact_geo_slide 0.005 \
  --lambda_hml_contact_geo_height 0.01 \
  --lambda_hml_contact_geo_vertical 0.01 \
  --lambda_hml_contact_geo_continuity 0.01 \
  --lambda_hml_contact_geo_smooth 0.001
