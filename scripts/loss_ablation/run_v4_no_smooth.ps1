$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

python -m train.train_mdm `
  --save_dir checkpoints/loss_v4_no_smooth `
  --arch plan_one `
  --lambda_hml_contact_geo_slide 0.01 `
  --lambda_hml_contact_geo_height 0.01 `
  --lambda_hml_contact_geo_vertical 0.01 `
  --lambda_hml_contact_geo_continuity 0.005
