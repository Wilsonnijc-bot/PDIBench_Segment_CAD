#!/usr/bin/env bash
set -euo pipefail

PDI_GPU_ROOT="${PDI_GPU_ROOT:-/root/autodl-tmp/pdi}"
FOUNDATIONPOSE_SOURCE="${PDI_FOUNDATIONPOSE_SOURCE:-$PDI_GPU_ROOT/cache/src/FoundationPose}"
WEIGHT_REPOSITORY="gpue/foundationpose-weights"
WEIGHT_REVISION="42d49e0633d245b3cf4dea6b1e7ec2b31d5b7654"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

download_file() {
  local relative_path="$1"
  local expected_sha256="$2"
  local destination="$FOUNDATIONPOSE_SOURCE/weights/$relative_path"
  local partial="$destination.partial"

  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]] && \
     echo "$expected_sha256  $destination" | sha256sum --check --status; then
    return
  fi
  curl --fail --location --retry 8 --retry-delay 2 --retry-all-errors \
    --continue-at - --output "$partial" \
    "$HF_ENDPOINT/$WEIGHT_REPOSITORY/resolve/$WEIGHT_REVISION/$relative_path"
  echo "$expected_sha256  $partial" | sha256sum --check
  mv "$partial" "$destination"
}

download_file \
  2023-10-28-18-33-37/model_best.pth \
  774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60 &
refiner_pid=$!
download_file \
  2024-01-11-20-02-45/model_best.pth \
  81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26 &
scorer_pid=$!
download_file 2023-10-28-18-33-37/config.yml \
  28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686 &
refiner_config_pid=$!
download_file 2024-01-11-20-02-45/config.yml \
  a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f &
scorer_config_pid=$!

wait "$refiner_pid"
wait "$scorer_pid"
wait "$refiner_config_pid"
wait "$scorer_config_pid"

echo "FoundationPose refiner and scorer weights are ready from $WEIGHT_REPOSITORY@$WEIGHT_REVISION"
