# MGMQ-PPO: Điều Khiển Đèn Giao Thông Thích Ứng

[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://www.python.org/)
[![SUMO](https://img.shields.io/badge/SUMO-1.24+-green.svg)](https://sumo.dlr.de/)
[![RLlib](https://img.shields.io/badge/Ray%2FRLlib-2.31+-orange.svg)](https://docs.ray.io/en/latest/rllib/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dự án này triển khai hệ thống **Điều khiển đèn giao thông thích ứng (Adaptive Traffic Signal Control - ATSC)** sử dụng kiến trúc **MGMQ (Multi-Layer Graph Masking Q-Learning)** kết hợp với thuật toán **PPO (Proximal Policy Optimization)**.

**Tác giả:** Son Dinh  
**Email:** ledinhsonx1td@gmail.com

---

## Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc MGMQ](#-kiến-trúc-mgmq)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu hình](#-cấu-hình)
- [Tham số huấn luyện](#-tham-số-huấn-luyện)
- [Tài liệu](#-tài-liệu)
- [Đóng góp](#-đóng-góp)
- [Giấy phép](#-giấy-phép)

---

## Tổng quan

Dự án bao gồm các thành phần chính:

1. **Module Tiền xử lý (GPI + FRAP):** Chuẩn hóa mạng lưới giao thông SUMO
   - **GPI (General Plug-In):** Chuẩn hóa làn đường với lane aggregation
   - **FRAP (Fine-grained Relation Attention Processing):** Chuẩn hóa pha đèn giao thông

2. **Kiến trúc MGMQ:** Mạng học sâu dựa trên đồ thị
   - **GAT (Graph Attention Network):** Nhúng thông tin giao lộ
   - **GraphSAGE + Bi-GRU:** Nhúng thông tin mạng lưới theo không gian-thời gian

3. **Thuật toán PPO:** Huấn luyện Reinforcement Learning với Actor-Critic

4. **Môi trường SUMO:** Tích hợp với trình mô phỏng giao thông SUMO

---

## Kiến trúc MGMQ

```
State (Observations)                                  
       │                                             
       ▼                                             
┌────────────────────────────────────────────────────┐
│                     MGMQ                           │
│  ┌─────────────┐     ┌──────────────────────────┐  │
│  │     GAT     │────▶│  GraphSAGE + Bi-GRU      │  │
│  │ (Intersection)    │     (Network)            │  │
│  └─────────────┘     └──────────────────────────┘  │
│         │                       │                  │
│         └──────────┬────────────┘                  │
│                    ▼                               │
│            Joint Embedding                         │
│                    │                               │
│         ┌─────────┴─────────┐                      │
│         ▼                   ▼                      │
│    Policy Head         Value Head                  │
│    (Actor)             (Critic)                    │
└────────────────────────────────────────────────────┘
       │                                             
       ▼                                             
Action (Traffic Signal Phase)                        
```

### Các thành phần chính:

| Thành phần | Mô tả | File |
|------------|-------|------|
| **GAT Layer** | Xử lý tương tác giữa các làn xe (Cooperation + Conflict graphs) | `src/models/gat_layer.py` |
| **GraphSAGE + Bi-GRU** | Tổng hợp thông tin không gian-thời gian trên mạng lưới | `src/models/graphsage_bigru.py` |
| **MGMQ Model** | Mô hình hoàn chỉnh với RLlib integration | `src/models/mgmq_model.py` |

---

## Yêu cầu hệ thống

### Phần mềm bắt buộc

| Phần mềm | Phiên bản | Ghi chú |
|----------|-----------|---------|
| Python | 3.10 - 3.12 | Khuyến nghị 3.11 |
| SUMO | 1.24.0+ | Cài đặt từ [sumo.dlr.de](https://sumo.dlr.de/) |
| Git | Mới nhất | Quản lý source code |

### Thư viện Python chính

| Thư viện | Phiên bản | Mô tả |
|----------|-----------|-------|
| `ray[rllib]` | ≥ 2.31.0 | Framework Reinforcement Learning |
| `torch` | ≥ 2.0.0 | Deep Learning backend |
| `gymnasium` | ≥ 1.1.1 | Gym environment API |
| `pettingzoo` | ≥ 1.25.0 | Multi-agent environments |
| `traci` | ≥ 1.24.0 | SUMO Python API |
| `numpy` | ≥ 1.24.0 | Numerical computing |
| `pandas` | ≥ 2.3.3 | Data analysis |

---

## Cài đặt

### 1. Clone dự án

```bash
git clone https://github.com/Sondinh2k3/MGMQ_for_ITS.git
cd MGMQ_for_ITS
```

### 2. Cài đặt SUMO

**Linux (Ubuntu/Debian):**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS:**
```bash
brew install sumo
```

**Windows:**
Tải từ [SUMO Downloads](https://sumo.dlr.de/docs/Downloads.php) và cài đặt.

**Thiết lập biến môi trường:**
```bash
export SUMO_HOME="/usr/share/sumo"  # Linux
export SUMO_HOME="/usr/local/opt/sumo/share/sumo"  # macOS
```

### 3. Cài đặt Poetry (Package Manager)

**Linux/macOS/WSL:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### 4. Cài đặt dependencies

```bash
# Tạo môi trường ảo và cài đặt thư viện
poetry install

# Kích hoạt môi trường ảo
poetry shell
```

### 5. Xác minh cài đặt

```bash
python verify_setup.py
```

---

## Cấu trúc thư mục

```
MGMQ_for_ITS/
├── scripts/                 # Script chính để chạy
│   ├── preprocess_network.py   # Tiền xử lý mạng lưới (GPI + FRAP)
│   ├── train_mgmq_ppo.py       # Huấn luyện MGMQ-PPO
│   └── eval_mgmq_ppo.py        # Đánh giá mô hình
│
├── src/                     # Mã nguồn chính
│   ├── models/              # Kiến trúc MGMQ
│   │   ├── gat_layer.py        # Graph Attention Network
│   │   ├── graphsage_bigru.py  # GraphSAGE + Bidirectional GRU
│   │   └── mgmq_model.py       # MGMQ Model cho RLlib
│   │
│   ├── environment/         # Môi trường RL
│   │   ├── rllib_utils.py      # RLlib utilities (SumoMultiAgentEnv)
│   │   └── drl_algo/
│   │       ├── env.py          # SumoEnvironment (Gym/PettingZoo)
│   │       ├── observations.py # Observation functions
│   │       └── traffic_signal.py # Traffic signal logic
│   │
│   ├── sim/                 # SUMO Simulator
│   │   ├── Sumo_sim.py         # SumoSimulator class
│   │   └── simulator_api.py    # Abstract API
│   │
│   ├── preprocessing/       # Module tiền xử lý
│   │   ├── frap.py             # FRAP phase standardization
│   │   └── standardizer.py     # Intersection standardization
│   │
│   └── config/              # Cấu hình
│       ├── model_config.yml    # Cấu hình chính (model, env, reward)
│       └── simulation.yml      # Tham số mô phỏng
│
├── network/                 # Mạng lưới giao thông SUMO
│   ├── grid4x4/                # Mạng lưới 4x4 (16 nút giao)
│   ├── PhuQuoc/                # Mạng lưới Phú Quốc
│   ├── zurich/                 # Mạng lưới Zurich
│   └── ...
│
├── docs/                    # Tài liệu
│   ├── MGMQ_Algorithm_Documentation.md
│   ├── Preprocessing_Module_Documentation.md
│   ├── Training_Testing_Pipeline.md
│   └── Parameter_Tuning_Guide.md
│
├── results_mgmq/            # Kết quả huấn luyện
│   └── mgmq_ppo_*/             # Checkpoints, logs, models
│
├── tests/                   # Unit tests
│   ├── test_env.py
│   ├── test_preprocessing.py
│   └── ...
│
├── tools/                   # Công cụ hỗ trợ
├── pyproject.toml              # Poetry dependencies
├── poetry.lock                 # Locked versions
├── verify_setup.py             # Kiểm tra cài đặt
└── README.md                   # Tài liệu này
```

---

## Hướng dẫn sử dụng

### Bước 1: Tiền xử lý mạng lưới

Chạy script để tạo file cấu hình JSON từ mạng lưới SUMO:

```bash
# Xử lý tất cả các nút giao trong mạng
python scripts/preprocess_network.py --network grid4x4

# Hoặc chỉ định các nút giao cụ thể
python scripts/preprocess_network.py --network grid4x4 --ts-ids A0 A1 B0 B1
```

**Output:** File `network/grid4x4/intersection_config.json` chứa:
- Thông tin làn đường chuẩn hóa
- Ánh xạ pha đèn
- Ma trận kết nối
- Cấu hình lane aggregation

### Bước 2: Huấn luyện mô hình

```bash
# Huấn luyện cơ bản
python scripts/train_mgmq_ppo.py --network grid4x4 --iterations 200

# Huấn luyện với nhiều workers
python scripts/train_mgmq_ppo.py \
    --network grid4x4 \
    --iterations 500 \
    --workers 4 \
    --learning-rate 3e-4

# Huấn luyện với GPU và early stopping
python scripts/train_mgmq_ppo.py \
    --network grid4x4 \
    --iterations 1000 \
    --workers 8 \
    --gpu \
    --reward-threshold -100 \
    --patience 50
```

**Output:** Thư mục `results_mgmq/mgmq_ppo_<network>_<timestamp>/` chứa:
- `checkpoints/` - Model checkpoints
- `logs/` - TensorBoard logs
- `config.json` - Training configuration
- `progress.csv` - Training metrics

### Bước 3: Theo dõi huấn luyện

```bash
# Mở TensorBoard
tensorboard --logdir results_mgmq/
```

Truy cập `http://localhost:6006` để xem metrics.

### Bước 4: Đánh giá mô hình

```bash
# Đánh giá checkpoint
python scripts/eval_mgmq_ppo.py \
    --checkpoint results_mgmq/mgmq_ppo_grid4x4_xxx/checkpoint_yyy \
    --network grid4x4 \
    --episodes 10

# Đánh giá với SUMO GUI
python scripts/eval_mgmq_ppo.py \
    --checkpoint results_mgmq/mgmq_ppo_grid4x4_xxx/checkpoint_yyy \
    --network grid4x4 \
    --gui

# Lưu kết quả ra file
python scripts/eval_mgmq_ppo.py \
    --checkpoint results_mgmq/mgmq_ppo_grid4x4_xxx/checkpoint_yyy \
    --network grid4x4 \
    --episodes 20 \
    --output evaluation_results.json
```

---

## Cấu hình

### File cấu hình chính: `src/config/model_config.yml`

```yaml
# Cấu hình mạng lưới
network:
  name: grid4x4
  net_file: network/grid4x4/grid4x4.net.xml
  route_file: network/grid4x4/grid4x4.rou.xml

# Cấu hình tiền xử lý
preprocessing:
  gpi:
    enabled: true
    lane_aggregation:
      standard_lanes_per_direction: 2
      missing_lane_strategy: zero    # "zero" hoặc "mask"
      merge_strategy: mean           # "sum", "mean", hoặc "max"
  frap:
    enabled: true
    standard_pattern: 4phase         # "4phase" hoặc "2phase"

# Cấu hình mô phỏng SUMO
environment:
  num_seconds: 8000       # Thời gian mô phỏng (giây)
  cycle_time: 90          # Chu kỳ đèn (giây)
  yellow_time: 3          # Thời gian đèn vàng
  min_green: 15           # Thời gian đèn xanh tối thiểu
  max_green: 60           # Thời gian đèn xanh tối đa
  use_gui: false          # Sử dụng SUMO GUI

# Danh sách nút giao được điều khiển
networks:
  grid4x4:
    ts_ids: [A0, A1, A2, A3, B0, B1, B2, B3, C0, C1, C2, C3, D0, D1, D2, D3]
```

---

## Tham số huấn luyện

### Tham số MGMQ Model

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `--gat-hidden-dim` | 256 | Kích thước ẩn của GAT |
| `--gat-output-dim` | 128 | Kích thước output của mỗi head GAT |
| `--gat-num-heads` | 4 | Số attention heads |
| `--graphsage-hidden-dim` | 256 | Kích thước ẩn của GraphSAGE |
| `--gru-hidden-dim` | 128 | Kích thước ẩn của Bi-GRU |
| `--dropout` | 0.3 | Dropout rate |

### Tham số PPO

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `--learning-rate` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--lambda` | 0.95 | GAE lambda |
| `--entropy-coeff` | 0.01 | Entropy coefficient |
| `--clip-param` | 0.2 | PPO clip parameter |

### Tham số Training

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `--iterations` | 200 | Số iterations huấn luyện |
| `--workers` | 4 | Số parallel workers |
| `--patience` | 50 | Early stopping patience |
| `--reward-threshold` | None | Ngưỡng reward để dừng |
| `--gpu` | false | Sử dụng GPU |
| `--history-length` | 4 | Độ dài window lịch sử quan sát |

### Ví dụ lệnh huấn luyện đầy đủ

```bash
python scripts/train_mgmq_ppo.py \
    --network grid4x4 \
    --iterations 500 \
    --workers 8 \
    --gpu \
    --gat-hidden-dim 128 \
    --gat-num-heads 8 \
    --graphsage-hidden-dim 128 \
    --gru-hidden-dim 64 \
    --learning-rate 1e-4 \
    --gamma 0.99 \
    --entropy-coeff 0.02 \
    --patience 100 \
    --history-length 5 \
    --reward-fn diff-waiting-time
```

---

## Tài liệu

Chi tiết hơn về từng module được trình bày trong thư mục `docs/`:

| Tài liệu | Nội dung |
|----------|----------|
| [MGMQ_Algorithm_Documentation.md](docs/MGMQ_Algorithm_Documentation.md) | Chi tiết kiến trúc MGMQ |
| [Preprocessing_Module_Documentation.md](docs/Preprocessing_Module_Documentation.md) | Module GPI và FRAP |
| [Training_Testing_Pipeline.md](docs/Training_Testing_Pipeline.md) | Quy trình huấn luyện và testing |
| [Parameter_Tuning_Guide.md](docs/Parameter_Tuning_Guide.md) | Hướng dẫn tuning hyperparameters |

---

## Chạy tests

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/test_env.py -v
pytest tests/test_preprocessing.py -v

# Chạy với coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Đóng góp

Đây là dự án mã nguồn mở. Mọi đóng góp đều được chào đón!

1. Fork dự án
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## Giấy phép

Dự án này được cấp phép theo **MIT License** - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## Tài liệu tham khảo

- **MGMQ Paper:** Multi-agent Graph-based Multi-scale Q-learning
- **Graph Attention Networks:** Veličković et al., ICLR 2018
- **GraphSAGE:** Hamilton et al., NeurIPS 2017  
- **PPO:** Proximal Policy Optimization, Schulman et al., 2017
- **SUMO:** https://sumo.dlr.de/docs/
- **RLlib:** https://docs.ray.io/en/latest/rllib/

---

## Liên hệ

- **Tác giả:** Son Dinh
- **Email:** ledinhsonx1td@gmail.com
- **GitHub:** [Sondinh2k3](https://github.com/Sondinh2k3)