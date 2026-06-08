# traffic-rl-bundle-tooling

Build-time CLI + library để package model bundle từ:
- Sim training output (policy.onnx, policy_meta.json, intersection_config.json)
- Operator's commissioning artifact (deployment_map.json)

Output: `bundle.zip` — đem upload MinIO → ai-algorithm-service tự pick up.

**Tách biệt rõ:**
- ❌ KHÔNG phải runtime — không cài vào Docker image của ai-algorithm-service
- ❌ KHÔNG phụ thuộc sim training framework
- ✅ Operator chạy 1 lần khi triển khai sang mạng thực

## Cài đặt

```bash
pip install -e ./bundle-tooling
# Dependency `traffic-rl-features` cài cùng từ ./traffic_rl_features (editable)
```

## CLI

```bash
build-bundle v2 \
  --sim-config       network/cologne3/intersection_config.json \
  --deployment-map   network/cologne3/deployment_map.json \
  --policy-onnx      <sim-output>/policy.onnx \
  --policy-meta      <sim-output>/policy_meta.json \
  --tenant-id        hcm_pilot \
  --version          v2026.05.15 \
  --output-zip       dist/cologne3-v2026.05.15.zip
```

## Library API

```python
from bundle_tooling import (
    DeploymentMap,
    build_v2_bundle_zip,
    validate as validate_deployment_map,
)

dm = DeploymentMap.model_validate_json(open("deployment_map.json").read())
# ...
```

## Cấu trúc

```
bundle_tooling/
├── deployment_map.py         # Pydantic schemas cho deployment_map.json
├── deployment_validator.py   # Cross-validate với sim_config
├── intersection_builder.py   # Translate sim → real IDs
├── packager.py               # build_v2_bundle_zip
├── cli.py                    # CLI entry point
└── feature_formula.py        # Re-export từ traffic_rl_features
```

## Quan hệ với các project khác

```
sim-training (mgmq) ──→ outputs (policy + intersection_config)
                                  │
                                  ▼
                        bundle-tooling (đây)  ◄── operator deployment_map.json
                                  │
                                  ▼
                              bundle.zip
                                  │
                                  ▼
                              MinIO
                                  │
                                  ▼
                    ai-algorithm-service (runtime)
```
