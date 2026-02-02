# Preprocessing Module Documentation

## Tổng quan

Module `preprocessing` thực hiện chuẩn hóa (standardization) cho hệ thống điều khiển đèn giao thông thông minh, dựa trên kiến trúc **GESA (Geometry-agnostic State-Action)**. Module này cho phép model học được **shared policy** có thể áp dụng cho các nút giao khác nhau về hình học và cấu hình đèn.

### Các thành phần chính:
1. **GPI (General Plug-In)** - `standardizer.py`: Chuẩn hóa hình học nút giao
2. **FRAP (Feature Relation Attention Processing)** - `frap.py`: Chuẩn hóa pha đèn tín hiệu

---

## 1. GPI Module - IntersectionStandardizer

### 1.1 Mục đích

GPI module chuyển đổi các hướng tiếp cận vật lý (physical approaches) của nút giao thành 4 hướng chuẩn: **North (N), East (E), South (S), West (W)**, bất kể hình học thực tế của mạng lưới.

### 1.2 Tại sao cần GPI?

```
Vấn đề: Các nút giao có hình học khác nhau
┌─────────────────────────────────────────────────────┐
│  Nút giao A (45°)        Nút giao B (0°)            │
│       ↗                        ↑                    │
│    ↙   ↘                    ←  +  →                 │
│       ↖                        ↓                    │
│                                                     │
│  → RL agent không thể học shared policy!            │
└─────────────────────────────────────────────────────┘

Giải pháp: Chuẩn hóa tất cả về N/E/S/W
┌─────────────────────────────────────────────────────┐
│  Nút giao A (chuẩn hóa)   Nút giao B (chuẩn hóa)   │
│         N                       N                   │
│      W  +  E                 W  +  E                │
│         S                       S                   │
│                                                     │
│  → RL agent học được shared policy!                 │
└─────────────────────────────────────────────────────┘
```

### 1.3 Input/Output

#### Constructor Input:
```python
IntersectionStandardizer(
    junction_id: str,        # SUMO junction/intersection ID
    data_provider: Any       # Interface cung cấp dữ liệu mạng lưới (optional)
)
```

**data_provider interface cần có:**
- `get_incoming_edges(junction_id) -> List[str]`: Trả về danh sách edge đi vào
- `get_lane_shape(lane_id) -> List[Tuple[float, float]]`: Trả về tọa độ các điểm của lane

#### Main Output:
```python
standard_map: Dict[str, Optional[str]] = {
    'N': "edge_north_id",   # hoặc None nếu không có
    'E': "edge_east_id",
    'S': "edge_south_id", 
    'W': "edge_west_id"
}
```

### 1.4 Thuật toán chuẩn hóa

```
┌─────────────────────────────────────────────────────────────────┐
│                    THUẬT TOÁN GPI                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BƯỚC 1: Lấy danh sách incoming edges                           │
│  ────────────────────────────────────                           │
│  edges = get_incoming_edges(junction_id)                        │
│  Ví dụ: ["edge_A", "edge_B", "edge_C", "edge_D"]               │
│                                                                 │
│  BƯỚC 2: Tính vector hướng cho mỗi edge                         │
│  ────────────────────────────────────────                       │
│  Với mỗi edge:                                                  │
│    - Lấy shape của lane đầu tiên (edge_0)                       │
│    - Lấy 2 điểm cuối: p1 (điểm trước), p2 (stop line)          │
│    - vector = normalize(p2 - p1)                                │
│                                                                 │
│       p1 ────────→ p2 (stop line)                               │
│              vector                                             │
│                                                                 │
│  BƯỚC 3: Chuyển vector thành góc (degrees)                      │
│  ─────────────────────────────────────────                      │
│  angle = arctan2(vector.y, vector.x) → [0°, 360°)              │
│                                                                 │
│              90° (S)                                            │
│                ↑                                                │
│     180° (E) ←─┼─→ 0° (W)                                       │
│                ↓                                                │
│             270° (N)                                            │
│                                                                 │
│  BƯỚC 4: Map góc → hướng chuẩn                                  │
│  ─────────────────────────────                                  │
│  ┌────────────────────────────┐                                │
│  │ Góc (degrees)  │  Hướng    │                                │
│  ├────────────────────────────┤                                │
│  │  225° - 315°   │    N      │  (vector chỉ xuống dưới)       │
│  │  135° - 225°   │    E      │  (vector chỉ sang trái)        │
│  │   45° - 135°   │    S      │  (vector chỉ lên trên)         │
│  │  315° - 45°    │    W      │  (vector chỉ sang phải)        │
│  └────────────────────────────┘                                │
│                                                                 │
│  BƯỚC 5: Xử lý conflict (nhiều edge cùng hướng)                 │
│  ──────────────────────────────────────────────                 │
│  Nếu nhiều edge map vào cùng 1 hướng:                           │
│    → Chọn edge có góc gần "góc lý tưởng" nhất                   │
│    Góc lý tưởng: N=270°, E=180°, S=90°, W=0°                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 Các methods quan trọng

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `map_intersection()` | None | `Dict[str, Optional[str]]` | Thực hiện chuẩn hóa và trả về mapping |
| `get_observation_mask()` | None | `np.ndarray` shape (4,) | Binary mask [N,E,S,W] - 1 nếu có, 0 nếu không |
| `get_standardized_edges()` | None | `List[Optional[str]]` | Danh sách edge theo thứ tự [N,E,S,W] |
| `get_edge_direction(edge_id)` | `str` | `Optional[str]` | Trả về hướng của edge (N/E/S/W) |
| `get_direction_edge(direction)` | `str` | `Optional[str]` | Trả về edge của hướng |
| `get_lanes_by_direction()` | None | `Dict[str, List[str]]` | Lanes grouped by direction |
| `export_config()` | None | `Dict[str, Any]` | Export cấu hình để lưu JSON |
| `load_config()` | `dict` | None | Load từ intersection_config.json |

### 1.6 Ví dụ sử dụng

```python
from src.preprocessing.standardizer import IntersectionStandardizer

# Khởi tạo
gpi = IntersectionStandardizer(junction_id="J1", data_provider=sumo_env)

# Thực hiện mapping
direction_map = gpi.map_intersection()
# Output: {'N': 'edge_from_north', 'E': 'edge_from_east', 'S': None, 'W': 'edge_from_west'}
# (S = None nghĩa là T-junction, không có hướng Nam)

# Lấy observation mask cho Hard Masking
mask = gpi.get_observation_mask()
# Output: array([1., 1., 0., 1.], dtype=float32) → có N, E, W; không có S

# Export để lưu
config = gpi.export_config()
```

---

## 2. FRAP Module - PhaseStandardizer

### 2.1 Mục đích

FRAP module chuẩn hóa các pha đèn tín hiệu thực tế thành **Standard Phase Pattern**, cho phép RL agent học phase selection dựa trên traffic demand cho từng movement cụ thể.

### 2.2 Khái niệm cơ bản

#### Movement Types (12 loại):
```
┌─────────────────────────────────────────────────────────┐
│                    MOVEMENTS                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Through (đi thẳng):    Left (rẽ trái):   Right (rẽ phải):
│  - NORTH_THROUGH (NT)   - NORTH_LEFT (NL)  - NORTH_RIGHT (NR)
│  - SOUTH_THROUGH (ST)   - SOUTH_LEFT (SL)  - SOUTH_RIGHT (SR)
│  - EAST_THROUGH (ET)    - EAST_LEFT (EL)   - EAST_RIGHT (ER)
│  - WEST_THROUGH (WT)    - WEST_LEFT (WL)   - WEST_RIGHT (WR)
│                                                         │
│  Minh họa NORTH_THROUGH và NORTH_LEFT:                  │
│                                                         │
│              ↑ (vào)                                    │
│              │                                          │
│              ↓ NT (đi thẳng xuống S)                    │
│         NL ←─┘ (rẽ trái sang W)                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Standard Phase Patterns:

**Pattern 4-pha (NEMA-like):**
```
┌────────────────────────────────────────────────────────────┐
│  Phase 0: NS Through     │   Phase 1: NS Left              │
│  ─────────────────────   │   ─────────────────             │
│         ↓                │          ↓                      │
│      W ─┼─ E             │       W ←─┼ E                   │
│         ↑                │          ↑                      │
│  {NT, ST}                │   {NL, SL}                      │
├────────────────────────────────────────────────────────────┤
│  Phase 2: EW Through     │   Phase 3: EW Left              │
│  ─────────────────────   │   ─────────────────             │
│         │                │          │                      │
│      W →┼← E             │       W ─┼← E                   │
│         │                │          ↓                      │
│  {ET, WT}                │   {EL, WL}                      │
└────────────────────────────────────────────────────────────┘
```

**Pattern 2-pha (đơn giản):**
```
┌────────────────────────────────────────────────────────────┐
│  Phase 0: All NS         │  Phase 1: All EW               │
│  ────────────────        │  ────────────────              │
│  {NT, ST, NL, SL}        │  {ET, WT, EL, WL}              │
└────────────────────────────────────────────────────────────┘
```

### 2.3 Input/Output

#### Constructor Input:
```python
PhaseStandardizer(
    junction_id: str,              # Traffic signal ID
    gpi_standardizer: Any,         # GPI module đã khởi tạo (để lấy direction)
    data_provider: Any             # Interface cung cấp signal program
)
```

**data_provider interface cần có:**
- `get_traffic_light_program(junction_id)`: Trả về signal program
- `get_controlled_links(junction_id)`: Trả về danh sách (from_lane, to_lane, via)

#### Data Classes:

```python
@dataclass
class Movement:
    movement_type: MovementType  # Loại movement (NT, ST, NL, ...)
    from_direction: str          # Hướng vào (N/E/S/W)
    to_direction: str            # Hướng ra (N/E/S/W)
    lanes: List[str]             # Danh sách lane phục vụ movement này
    is_protected: bool           # True = protected, False = permitted

@dataclass  
class Phase:
    phase_id: int                      # ID của phase trong signal program
    movements: Set[MovementType]       # Các movements được phục vụ trong phase này
    duration_range: Tuple[int, int]    # (min, max) duration in seconds
    is_yellow: bool                    # True nếu là yellow phase
    state: str                         # Phase state string (e.g., "GGGrrr")
    duration: float                    # Phase duration
    green_indices: List[int]           # Indices of green signals
```

### 2.4 Thuật toán chuẩn hóa Phase

```
┌─────────────────────────────────────────────────────────────────┐
│               THUẬT TOÁN FRAP (configure)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BƯỚC 1: Lấy signal program và controlled links                 │
│  ──────────────────────────────────────────────                 │
│  program = get_traffic_light_program(junction_id)               │
│  controlled_links = get_controlled_links(junction_id)           │
│                                                                 │
│  Ví dụ program.phases:                                          │
│  [                                                              │
│    Phase(state="GGGrrr", duration=30),   # Green cho link 0,1,2 │
│    Phase(state="yyyrrr", duration=3),    # Yellow               │
│    Phase(state="rrrGGG", duration=30),   # Green cho link 3,4,5 │
│    Phase(state="rrryyy", duration=3),    # Yellow               │
│  ]                                                              │
│                                                                 │
│  BƯỚC 2: Trích xuất green phases (bỏ qua yellow)                │
│  ───────────────────────────────────────────────                │
│  Với mỗi phase:                                                 │
│    - Kiểm tra 'y' trong state → skip nếu yellow                 │
│    - Lấy green_indices = vị trí các ký tự 'G' trong state       │
│                                                                 │
│  BƯỚC 3: Map controlled_links → MovementType                    │
│  ──────────────────────────────────────────────                 │
│  Với mỗi link (from_lane, to_lane, via):                        │
│    1. Lấy from_edge = from_lane.rsplit('_', 1)[0]               │
│    2. Lấy to_edge = to_lane.rsplit('_', 1)[0]                   │
│    3. Dùng GPI để lấy hướng chuẩn:                              │
│       from_dir = gpi.get_edge_direction(from_edge)              │
│       to_dir = gpi.get_edge_direction(to_edge)                  │
│    4. Infer movement type từ direction change:                  │
│                                                                 │
│       ┌──────────────────────────────────────────┐              │
│       │ from_dir │ diff=2 │ diff=1 │ diff=3      │              │
│       │──────────│────────│────────│───────────--│              │
│       │    N     │   NT   │   NR   │   NL        │              │
│       │    S     │   ST   │   SR   │   SL        │              │
│       │    E     │   ET   │   ER   │   EL        │              │
│       │    W     │   WT   │   WR   │   WL        │              │
│       └──────────────────────────────────────────┘              │
│       (diff = (to_idx - from_idx) % 4)                          │
│                                                                 │
│  BƯỚC 4: Xác định movements cho mỗi phase                       │
│  ────────────────────────────────────────────                   │
│  Với mỗi green phase:                                           │
│    phase_movements = {link_movements[i] for i in green_indices} │
│                                                                 │
│  BƯỚC 5: Tạo standard phase mapping                             │
│  ──────────────────────────────────                             │
│  Với mỗi actual phase:                                          │
│    - Tính overlap với từng standard phase                       │
│    - Map actual → standard có overlap lớn nhất                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.5 Inference Movement Type Logic

```
Direction order: N=0, E=1, S=2, W=3

Công thức: diff = (to_idx - from_idx) % 4

┌─────────────────────────────────────────────────────────────┐
│  diff = 2 → THROUGH (đi thẳng, đối diện)                    │
│  diff = 1 → RIGHT (rẽ phải, theo chiều kim đồng hồ)         │
│  diff = 3 → LEFT (rẽ trái, ngược chiều kim đồng hồ)         │
│  diff = 0 → U-TURN (không hỗ trợ)                           │
└─────────────────────────────────────────────────────────────┘

Ví dụ:
  N → S: from=0, to=2, diff=(2-0)%4=2 → NORTH_THROUGH
  N → E: from=0, to=1, diff=(1-0)%4=1 → NORTH_RIGHT
  N → W: from=0, to=3, diff=(3-0)%4=3 → NORTH_LEFT
  E → W: from=1, to=3, diff=(3-1)%4=2 → EAST_THROUGH
```

### 2.6 Các methods quan trọng

| Method | Input | Output | Mô tả |
|--------|-------|--------|-------|
| `configure()` | None | None | Thực hiện cấu hình FRAP (gọi 1 lần) |
| `load_config()` | `Dict` | None | Load từ intersection_config.json |
| `get_phase_demand_features()` | `density_by_dir, queue_by_dir` | `np.ndarray` (8,) | Feature vector cho 4 phases × 2 metrics |
| `standardize_action()` | `np.ndarray` (4,) | `np.ndarray` (num_phases,) | Convert standard → actual action |
| `get_movement_mask()` | None | `np.ndarray` (8,) | Binary mask cho 8 movements |
| `get_phase_mask()` | None | `np.ndarray` (4,) | Binary mask cho 4 standard phases |

### 2.7 Ví dụ sử dụng

```python
from src.preprocessing.standardizer import IntersectionStandardizer
from src.preprocessing.frap import PhaseStandardizer

# Bước 1: Khởi tạo GPI trước
gpi = IntersectionStandardizer(junction_id="J1", data_provider=env)
gpi.map_intersection()

# Bước 2: Khởi tạo FRAP với GPI
frap = PhaseStandardizer(
    junction_id="J1",
    gpi_standardizer=gpi,
    data_provider=env
)

# Bước 3: Configure
frap.configure()

# Bước 4: Sử dụng
# Lấy phase demand features
density = {'N': 0.5, 'E': 0.3, 'S': 0.4, 'W': 0.2}
queue = {'N': 0.3, 'E': 0.1, 'S': 0.2, 'W': 0.1}
features = frap.get_phase_demand_features(density, queue)
# Output shape: (8,) - [phase0_density, phase1_density, phase2_density, phase3_density,
#                       phase0_queue, phase1_queue, phase2_queue, phase3_queue]

# Convert standard action → actual
standard_action = np.array([0.4, 0.1, 0.4, 0.1])  # 4 standard phases
actual_action = frap.standardize_action(standard_action)
# Output shape: (num_actual_phases,)

# Lấy masks
movement_mask = frap.get_movement_mask()  # (8,)
phase_mask = frap.get_phase_mask()        # (4,)
```

---

## 3. Tích hợp GPI + FRAP

### 3.1 Pipeline hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                           │
│  │  SUMO Network   │                                           │
│  │  (.net.xml)     │                                           │
│  └────────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │                GPI Module                            │       │
│  │  ────────────────────────────────────────────────── │       │
│  │  Input: junction geometry (edges, lane shapes)       │       │
│  │  Output: direction_map {N/E/S/W → edge_id}          │       │
│  │          observation_mask [1,1,0,1]                  │       │
│  └────────────────────────┬────────────────────────────┘       │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │               FRAP Module                            │       │
│  │  ────────────────────────────────────────────────── │       │
│  │  Input: GPI directions + signal program             │       │
│  │  Output: actual_to_standard mapping                  │       │
│  │          phase_demand_features                       │       │
│  │          movement/phase masks                        │       │
│  └────────────────────────┬────────────────────────────┘       │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │            Standardized Features                     │       │
│  │  ────────────────────────────────────────────────── │       │
│  │  • Fixed-size observation: [N, E, S, W] features    │       │
│  │  • Fixed-size action: 4 standard phases             │       │
│  │  • Masks for invalid directions/phases              │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Lợi ích của Standardization

| Aspect | Không chuẩn hóa | Có GPI + FRAP |
|--------|-----------------|---------------|
| **Observation size** | Thay đổi theo junction | Fixed size (4 directions) |
| **Action size** | Thay đổi theo signal program | Fixed size (4 phases) |
| **Transfer Learning** | Không thể | Có thể |
| **Shared Policy** | Không | Có |
| **T-junction xử lý** | Custom code | Automatic masking |

### 3.3 Hard Masking với Preprocessing

```python
# Sử dụng masks từ preprocessing cho Hard Masking trong RL

# 1. Direction mask (từ GPI) - cho observation
direction_mask = gpi.get_observation_mask()
# [1, 1, 0, 1] → có N, E, W; không có S

# 2. Phase mask (từ FRAP) - cho action  
phase_mask = frap.get_phase_mask()
# [1, 1, 0, 0] → chỉ có phase 0, 1; không có phase 2, 3

# Trong model:
# - Observation features của missing direction = 0
# - Action logits của missing phase = -∞ (masked)
```

---

## 4. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SUMO Environment                                                       │
│   ─────────────────                                                      │
│   │                                                                      │
│   ├─→ get_incoming_edges(junction_id) ─┐                                │
│   │                                     │                                │
│   ├─→ get_lane_shape(lane_id) ─────────┼─→ GPI.map_intersection()       │
│   │                                     │           │                    │
│   │                                     │           ▼                    │
│   │                              direction_map = {N: edge1, E: edge2, ...}
│   │                              observation_mask = [1, 1, 0, 1]         │
│   │                                                 │                    │
│   │                                                 ▼                    │
│   ├─→ get_traffic_light_program(tls_id) ─┐         │                    │
│   │                                       │         │                    │
│   ├─→ get_controlled_links(tls_id) ──────┼─→ FRAP.configure()           │
│   │                                       │    (uses GPI directions)     │
│   │                                       │           │                  │
│   │                                       │           ▼                  │
│   │                                actual_to_standard = {0: 0, 1: 2}    │
│   │                                phase_to_movements = {0: {NT,ST}, ...}│
│   │                                movement_mask = [1,1,1,1,1,0,1,0]    │
│   │                                phase_mask = [1, 0, 1, 0]            │
│   │                                                                      │
│   │   Runtime (mỗi step)                                                │
│   │   ─────────────────                                                 │
│   │                                                                      │
│   ├─→ get_density_by_detector() ─┐                                      │
│   │                               │                                      │
│   ├─→ get_queue_by_detector() ───┼─→ FRAP.get_phase_demand_features()  │
│   │                               │           │                          │
│   │                               │           ▼                          │
│   │                     standardized_features = np.array([...], shape=8)│
│   │                                                                      │
│   │   Action                                                             │
│   │   ──────                                                            │
│   │   standard_action = policy(standardized_features)  # shape (4,)     │
│   │           │                                                          │
│   │           ▼                                                          │
│   │   actual_action = FRAP.standardize_action(standard_action)          │
│   │           │                                         # shape (num_phases,)
│   │           ▼                                                          │
│   └─→ TrafficSignal.set_next_phase(actual_action)                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. intersection_config.json Format

Khi chạy `preprocess_network.py`, file `intersection_config.json` được tạo ra với format:

```json
{
  "network_name": "grid4x4",
  "intersections": {
    "A0": {
      "gpi": {
        "direction_map": {"N": "A0_N", "E": "A0_E", "S": "A0_S", "W": "A0_W"},
        "observation_mask": [1.0, 1.0, 1.0, 1.0],
        "lanes_by_direction": {
          "N": ["A0_N_0", "A0_N_1"],
          "E": ["A0_E_0", "A0_E_1"],
          "S": ["A0_S_0", "A0_S_1"],
          "W": ["A0_W_0", "A0_W_1"]
        }
      },
      "frap": {
        "num_phases": 4,
        "num_green_phases": 4,
        "actual_to_standard": [0, 1, 2, 3],
        "standard_to_actual": [[0], [1], [2], [3]],
        "phase_mask": [1.0, 1.0, 1.0, 1.0],
        "movement_mask": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "phases": [
          {"phase_id": 0, "state": "GGGrrrGGGrrr", "duration": 30.0, "movements": ["NT", "ST"]}
        ]
      }
    }
  }
}
```

---

## 6. Lưu ý quan trọng

### 6.1 Thứ tự khởi tạo
```python
# 1. GPI phải được khởi tạo và map trước
gpi = IntersectionStandardizer(junction_id, data_provider)
gpi.map_intersection()  # PHẢI gọi

# 2. FRAP cần GPI
frap = PhaseStandardizer(junction_id, gpi_standardizer=gpi, data_provider)
frap.configure()  # PHẢI gọi
```

### 6.2 T-junction và Irregular Intersections
- GPI tự động xử lý missing directions (set None, mask = 0)
- FRAP map phases dựa trên movements có sẵn
- RL agent sử dụng masks để ignore invalid directions/phases

### 6.3 Caching và Load từ Config
- Cả GPI và FRAP đều có `load_config()` method
- Trong production, load từ `intersection_config.json` để tránh TraCI calls
- Chỉ cần chạy `preprocess_network.py` một lần cho mỗi network mới

### 6.4 Scripts

```bash
# Preprocessing một network
python scripts/preprocess_network.py --network grid4x4

# Output: network/grid4x4/intersection_config.json
```

---

## 7. References

1. **GESA Architecture**: "Geometry-agnostic State-Action for Traffic Signal Control"
2. **FRAP Concept**: IntelliLight - "Feature Relation Attention Processing"
3. **NEMA Phases**: Standard traffic signal phase patterns in US
