# traffic_rl_features

Shared feature preprocessing cho **sim training + runtime inference** trong dự án RL cho điều khiển đèn tín hiệu.

## Vấn đề giải quyết

Khi training RL trong SUMO, sim env đọc detector e1/e2 và áp **công thức** để build observation. Khi deploy ra thực tế, runtime service nhận API gửi `(occupancy, speed)` từ camera/loop và phải áp **cùng công thức** để policy thấy distribution observation giống lúc train.

Nếu hai bên viết tay (logic copy-paste), drift là chuyện không tránh khỏi — đây là package **dùng chung** đảm bảo single source of truth.

## Cài đặt

```bash
# Editable install vào virtualenv của cả service và sim trainer
pip install -e ./Service-ai/traffic_rl_features
```

## Public API

```python
from traffic_rl_features import (
    FeatureSpec,       # Schema cho feature_formula.json
    FeatureBuilder,    # Compute N-channel feature từ road vars
    default_spec,      # Fallback 4-channel spec
    PACKAGE_VERSION,   # Contract version, embed vào bundle manifest
)
```

## Workflow

### 1. Sim trainer (SUMO)

```python
from traffic_rl_features import FeatureSpec, FeatureBuilder
from traffic_rl_features.sim_helpers import (
    sumo_detector_to_road_vars,
    road_static_from_sumo_lane,
)

# Lúc init env: build roads_static từ SUMO network XML
roads_static = {
    "4999334": road_static_from_sumo_lane(num_lanes=1, lane_length_meters=142.34),
    "-241660955#6": road_static_from_sumo_lane(num_lanes=2, lane_length_meters=196.34),
    # ...
}

# Load spec đã commission (cùng file embed vào bundle)
spec = FeatureSpec.from_file("network/cologne3/feature_formula.json")
builder = FeatureBuilder(spec=spec, roads_static=roads_static)

# Mỗi step: đọc detector, convert → vars → eval
for road_id, det in env.detectors.items():
    occ_speed = sumo_detector_to_road_vars(
        e1_occupancy_percent=det.e1_occupancy(),
        e2_occupancy_percent=det.e2_occupancy(),
        e1_mean_speed_ms=det.e1_mean_speed(),
    )
    feat = builder.compute(road_id, **occ_speed)  # ndarray(4,)
```

### 2. Runtime service (production)

```python
from traffic_rl_features import FeatureSpec, FeatureBuilder

spec = FeatureSpec.from_file(bundle_root / "feature_formula.json")
builder = FeatureBuilder(spec=spec, roads_static=roads_static_from_bundle)

for road in cross.roads:
    feat = builder.compute(
        road.id,
        occupancy=road.occupancySpace,
        speed=road.averageSpeed,
    )  # ndarray(4,)
```

Cùng `spec` + cùng `roads_static` → `builder.compute` ra cùng kết quả. Không drift.

## Định nghĩa formula

`feature_formula.json` example:

```json
{
  "channels": ["density", "queue", "occupancy", "speed"],
  "formulas": {
    "density":   "clip(occupancy / 100.0, 0.0, 1.0)",
    "queue":     "clip(occupancy/100 * lanes * length / 7.5, 0.0, lanes * length / 7.5)",
    "occupancy": "clip(occupancy / 100.0, 0.0, 1.0)",
    "speed":     "clip(speed / max(speed_design, 50.0), 0.0, 1.5)"
  }
}
```

Biến cho phép trong expression:

| Tên | Đơn vị | Nguồn |
|---|---|---|
| `occupancy` | % [0, 100] | Realtime — `Road.occupancySpace` ở runtime / detector e1 hoặc e2 ở sim |
| `speed` | km/h | Realtime — `Road.averageSpeed` ở runtime / detector mean_speed × 3.6 ở sim |
| `lanes` | int | Static — từ bundle / SUMO network |
| `length` | m | Static |
| `speed_design` | km/h | Static |
| `saturation_flow` | veh/h | Static |

Operator/function: `+ - * / ** % //`, `==, !=, <, >, <=, >=`, `and`, `or`, ternary `x if c else y`, `min(...)`, `max(...)`, `abs(x)`, `clip(x, lo, hi)`.

Cấm: import, attribute access, subscript, lambda, comprehension, gọi hàm khác.

## Versioning

`PACKAGE_VERSION` follow SemVer:
- **MAJOR**: contract không tương thích — bump khi đổi syntax/vars/default. Bundle manifest có `feature_pkg_version` major khác runtime → fail-fast.
- **MINOR**: thêm tính năng backward-compatible.
- **PATCH**: bug fix evaluator.

Sim trainer + service nên pin cùng MAJOR version.

## Test

```bash
cd Service-ai/traffic_rl_features
pytest tests/ -v
```
