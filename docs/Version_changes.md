# Nhật Ký Thay Đổi & Đánh Giá Hiệu Quả

> **Dự án:** Hệ thống điều khiển đèn giao thông thích ứng (GPI + FRAP + MGMQ + PPO)  
> **Ngày bắt đầu:** 2026-01-17  
> **Mô tả:** Ghi lại lịch sử thay đổi và kết quả đánh giá hiệu quả qua từng phiên bản/thử nghiệm.

---

## 📌 Mục Lục
- [Nhật Ký Thay Đổi (Changelog)](#-nhật-ký-thay-đổi-changelog)
- [Đánh Giá Hiệu Quả (Experiments)](#-đánh-giá-hiệu-quả-experiments)
- [Ghi Chú Chung](#-ghi-chú-chung)

---

## Nhật Ký Thay Đổi (Changelog)

### [v1.0.0] - 2026-01-17
#### ✨ Thêm mới (Added)
- Tạo khung dự án
- Phiên bản tam thời chạy được

#### 🔄 Thay đổi (Changed)
- Tăng `train_batch_size` lên 4096 (trước là 320) để đảm bảo đủ mẫu cho PPO update.
- Tăng `rollout_fragment_length` lên 32 (trước là 5) để giảm overhead sync.
- Tăng `minibatch_size` lên 128.
- Tăng `sample_timeout_s` lên 3600s (1h) trong RLlib config.
- Tăng `_wall_timeout` trong `SumoSimulator.step` lên 300s để tránh crash worker khi máy lag.

#### 🐛 Sửa lỗi (Fixed)
- Fix lỗi NaN reward do số lượng episode hoàn thành = 0 (do batch size quá nhỏ và worker bị crash).

#### 🗑️ Loại bỏ (Removed)
- Version đầu, chưa cập nhật

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `path/to/file.py` | Modified | Mô tả thay đổi |
| `path/to/new_file.py` | Added | Mô tả file mới |

---

### [v1.1.0] - 2026-01-18
#### ✨ Thêm mới (Added)
- Episode-based training: Cập nhật weights sau mỗi episode hoàn thành thay vì chờ đủ batch size

#### 🔄 Thay đổi (Changed)
- **rollout_fragment_length**: 8 → `"auto"` (tự động tính dựa trên batch size)
- **batch_mode**: default → `"complete_episodes"` (chờ episode hoàn thành)
- **train_batch_size**: 4096 → **1424** (= 89 env steps × 16 agents, khớp 1 episode)
- **minibatch_size**: 128 → 256
- **num_sgd_iter**: 10 → 4 (giảm SGD iterations để update nhanh hơn)
- **step-length**: 0.1 → 0.5 (tăng tốc simulation 2x)

#### 🐛 Sửa lỗi (Fixed)
- Fix vấn đề training quá chậm (~8.5h/iteration) do:
  - `train_batch_size` quá lớn (4096) so với reference (128)
  - `rollout_fragment_length` không phù hợp với episode length
  - Phải đợi quá nhiều samples trước khi update weights

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `scripts/train_mgmq_ppo.py` | Modified | Episode-based config: rollout_fragment_length="auto", batch_mode="complete_episodes" |
| `src/config/model_config.yml` | Modified | Giảm train_batch_size từ 2048 xuống 512 |

---

### [v1.1.1] - 2026-01-18
#### ✨ Thêm mới (Added)
- Chưa thêm gì mới

#### 🔄 Thay đổi (Changed)
- Không thay đổi gì

#### 🐛 Sửa lỗi (Fixed)
- Fix vấn đề không đồng nhất về các tham số, cấu hình mô phỏng giữa chạy baseline và chạy đánh giá thuật toán.

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `scripts/train_mgmq_ppo.py` | Modified | Thêm các tham số cấu hình sao cho match với file .sumocfg của network |
| `scripts/eval_mgmq_ppo.py` | Modified | Thêm các tham số cấu hình sao cho match với file .sumocfg của network|

---

### [v1.1.2] - 2026-01-23
#### ✨ Thêm mới (Added)
- Chưa thêm gì mới

#### 🔄 Thay đổi (Changed)
- Thêm giới hạn biên cho giá trị log(std): [Xem giải thích chi tiết](Explanation_Log_Std.md)

#### 🐛 Sửa lỗi (Fixed)
- Sửa lại lớp đồ thị mạng lưới: GraphSAGE + BiGRU

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `graphsage_bigru.py` | Modified | Sửa lại cấu trúc của GraphSAGE -> GraphSAGE nâng cao, và BiGRU lúc này chỉ nhằm mục đích tổng hợp thông tin cho output của GraphSAGE |
| `mgmq_model.py` | Modified | Thêm giới hạn cho log(std)|

---

### [v1.2.0] - 2026-01-23
#### ✨ Thêm mới (Added)
- **Directional Adjacency Matrix**: Tạo module mới để xây dựng ma trận kề có hướng từ file SUMO .net.xml
  - Phân loại neighbor theo 4 hướng chuẩn (North, East, South, West) dựa trên tọa độ địa lý
  - Tính toán góc vector từ node A đến neighbor B để xác định hướng chính xác
  - Hỗ trợ cả ma trận kề đơn giản (backward compatible)

#### 🔄 Thay đổi (Changed)
- **GraphSAGE Logic**: Sửa lại logic neighbor exchange để sử dụng đúng mask hướng:
  - `in_north = torch.bmm(mask_north, g_south)` — Đầu vào cổng Bắc từ đầu ra hướng Nam của neighbor phía Bắc
  - `in_east = torch.bmm(mask_east, g_west)` — Đầu vào cổng Đông từ đầu ra hướng Tây của neighbor phía Đông
  - Tương tự cho hướng Nam và Tây
  - **Trước đây**: Sử dụng một ma trận kề duy nhất cho tất cả hướng → Nhầm lẫn thông tin từ các hướng khác nhau
  - **Bây giờ**: Sử dụng ma trận riêng cho từng hướng → Đúng vật lý, chính xác hơn
- **DirectionalGraphSAGE.forward()**: Nhận đầu vào `adj_directions: [Batch, 4, N, N] or [4, N, N]`
- **GraphSAGE_BiGRU.forward()**: Cập nhật chữ ký hàm để nhận `adj_directions`
- **TemporalGraphSAGE_BiGRU.forward()**: Cập nhật để nhận và xử lý `adj_directions` đúng cách
- **build_network_adjacency()**: 
  - Thêm tham số `directional: bool = True`
  - Tính toán góc hướng từ tọa độ junction trong file .net.xml
  - Trả về tensor `[4, N, N]` khi `directional=True`
- **MGMQEncoder**: 
  - Cập nhật để nhận và xử lý ma trận kề `[4, N, N]`
  - Tự động expand ma trận kề đơn giản thành ma trận có hướng nếu cần
- **LocalTemporalMGMQEncoder._build_star_adjacency()**: Trả về `[B, 4, N, N]` thay vì `[B, N, N]`

#### 🐛 Sửa lỗi (Fixed)
- **Lỗi logic vật lý**: Trước đây neighbor exchange không phân biệt hướng, dẫn đến nhầm lẫn thông tin spatial
- **Ma trận kề không phản ánh topology**: Bây giờ ma trận kề chứa đúng thông tin hướng từ tọa độ địa lý

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `src/preprocessing/graph_builder.py` | Added | Module mới: xây dựng directional adjacency matrix từ SUMO |
| `src/models/graphsage_bigru.py` | Modified | Cập nhật forward để nhận `adj_directions [4,N,N]` thay vì `adj [N,N]` |
| `src/models/mgmq_model.py` | Modified | Cập nhật `build_network_adjacency()` để tạo ma trận có hướng, cập nhật `MGMQEncoder` |
| `src/preprocessing/__init__.py` | Modified | Export các hàm mới từ `graph_builder.py` |

#### 💡 Nhận xét Kỹ Thuật
- **Vấn đề được giải quyết**: Trước đây mô hình không tận dụng được thông tin topology có hướng của mạng giao thông, tất cả neighbor được xử lý như nhau
- **Cải thiện đạt được**: 
  - Logic neighbor exchange giờ đây tuân theo vật lý thực tế (xe từ phía Bắc chảy vào cổng Bắc)
  - Mô hình có thể học được các pattern khác biệt giữa các hướng
  - Embedding network sẽ chứa đúng thông tin spatial relationship
- **Backward Compatibility**: Vẫn hỗ trợ ma trận kề đơn giản, tự động mở rộng thành ma trận có hướng

---

### [v1.2.1] - 2026-01-23
#### ✨ Thêm mới (Added)
- Không có

#### 🔄 Thay đổi (Changed)
- **Code Quality Improvements**: Clean code và cải thiện documentation
  - **DirectionalGraphSAGE.forward()**: 
    - Thêm input validation với assert statements
    - Cải thiện docstring với chi tiết về input/output shapes
    - Thêm section comments rõ ràng (Step 1, 2, 3, 4)
  - **GraphSAGE_BiGRU**: 
    - Cải thiện docstring với giải thích rõ về API compatibility
    - Thêm type hints đầy đủ
  - **TemporalGraphSAGE_BiGRU**: 
    - Cải thiện docstring với giải thích về pipeline (Spatial -> Temporal -> Pooling)
    - Thêm section comments cho từng bước xử lý
  - **LocalTemporalMGMQEncoder._build_star_adjacency()**: 
    - Cải thiện docstring với giải thích chi tiết về node indexing và edge logic
    - Thêm ASCII art cho node layout

#### 🐛 Sửa lỗi (Fixed)
- Sửa comment sai trong mgmq_model.py: `[B, 1+K, 1+K]` → `[B, 4, 1+K, 1+K]`

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `src/models/graphsage_bigru.py` | Modified | Clean code: improved docstrings, type hints, section comments |
| `src/models/mgmq_model.py` | Modified | Fixed comment, improved _build_star_adjacency docstring |
- **Test Results**: ✓ DirectionalGraphSAGE test passed | ✓ TemporalGraphSAGE_BiGRU test passed | ✓ build_network_adjacency test passed

---

### [v1.2.2] - 2026-01-27
#### ✨ Thêm mới (Added)
- Không có

#### 🔄 Thay đổi (Changed)
- **Observation Structure**: Chuyển đổi cấu trúc vector quan sát từ **Feature-major** sang **Lane-major**.
  - **Trước đây**: `[All_Densities, All_Queues, All_Occupancies, All_Speeds]`
  - **Bây giờ**: `[Lane0_Feats, Lane1_Feats, ..., Lane11_Feats]`
  - **Lý do**: Model GAT (`mgmq_model.py`) sử dụng `.view(-1, 12, 4)` để tách đặc trưng cho từng lane. Với cấu trúc cũ, Lane 0 nhận nhầm 4 giá trị density của 4 lane đầu tiên thay vì 4 đặc trưng của chính nó.
  - **Ảnh hưởng**: Thay đổi ý nghĩa của input features. **BẮT BUỘC** phải train lại model mới, model cũ sẽ hoạt động sai lệch.

#### 🐛 Sửa lỗi (Fixed)
- **Critical Bug Fix**: Sửa lỗi mismatch giữa `observations.py` và `mgmq_model.py`. Đảm bảo GAT layer nhận đúng đặc trưng vật lý của từng lane.
- **Baseline Evaluation**: Sửa lỗi `eval_baseline_reward.py` để dùng `fixed_ts=True` và `SumoMultiAgentEnv` chuẩn, đảm bảo metrics so sánh (steps, reward) nhất quán với training.

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| `src/environment/drl_algo/observations.py` | Modified | Reorder observation vector to Lane-major |
| `tools/eval_baseline_reward.py` | Modified | Rewrite to match eval_mgmq_ppo.py structure |

---

<!-- TEMPLATE CHO CHANGELOG MỚI - Copy phần này khi thêm version mới -->
<!--
### [vX.X.X] - YYYY-MM-DD
#### ✨ Thêm mới (Added)
- 

#### 🔄 Thay đổi (Changed)
- 

#### 🐛 Sửa lỗi (Fixed)
- 

#### 🗑️ Loại bỏ (Removed)
- 

#### 📁 Files thay đổi
| File | Loại | Mô tả ngắn |
|------|------|-----------|
| | | |

---
-->

---

## Đánh Giá Hiệu Quả (Experiments)

### Experiment #001 - 2026-01-17
**Mục tiêu:** Thử nghiệm đánh giá phiên bản đầu tiên: đánh giá về các đồ thị, về hiệu quả sau khi training.

#### 🔧 Tham số (Parameters)
| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `learning_rate` | 0.0003 | |
| `batch_size` | 4096 | |
| `gamma` | 0.99 | |
| `num_episodes` | 10 | |
| `network_arch` | [256, 256] | |

#### 📈 Kết quả (Results)
| Metric | Giá trị | So sánh với baseline |
|--------|---------|---------------------|
| Mean Reward | 150.5 | +20% |
| Episode Length | 200 | -10% |
| Convergence Step | 5000 | - |
| Training Time | 2h 30m | - |

#### 📉 Biểu đồ (nếu có)
<!-- Thêm link hoặc embed hình ảnh -->
<!-- ![Tên biểu đồ](path/to/chart.png) -->

#### 💡 Nhận xét & Kết luận
- Điểm mạnh:
  - 
- Điểm yếu/Vấn đề:
  - Các lớp hidden layer của policy và value đang không được truyền đúng từ file config => cần sửa lại
  - Các giá trị tính toán ra bằng 0 hoặc NaN? => Nguyên nhân có lẽ là do: RLlib tính toán các giá trị này (ví dụ: episode_reward_mean) khi một episode hoàn thành. TUy nhiên, do cấu hình num_second là 8000, trong khi mỗi iteration chỉ lấy mẫu được 40-50 bước => Cần chạy hàng trăm iteration mới xong 1 episode => khi đó mẫu số (số episode = 0) nên việc chia cho số episode để tính trung bình sẽ ra NaN.
  - train_batch_size đang quá thấp (320) và rollout_fragment_length cũng quá thấp (40) => Mỗi worker chỉ chạy 5 bước rồi dừng để gửi dữ liệu về.
  - Nói chung vấn đề đang là workers bị crash giữa chừng vì nhiều lý do.
- Kết luận:
  - 
- Hướng cải tiến tiếp theo:
  - 

### Experiment #002 - 2026-01-19
**Mục tiêu:** 

#### 🔧 Tham số (Parameters)
| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `learning_rate` |0.0003 | |
| `batch_size` |1424 | |
| `gamma` |0.99 | |
| `num_episodes` |20 | |
| | | |

#### 📈 Kết quả (Results)
| Metric | Giá trị | So sánh với baseline |
|--------|---------|---------------------|
| Mean Reward |-275 -> -245 | |
| Episode Length |89 | |
| Convergence Step | | |
| Training Time |3h | |

#### 📉 Biểu đồ (nếu có)

##### So sánh tổng hợp (Before vs After)
| Biểu đồ | Mô tả |
|---------|-------|
| ![Congestion Overview Combined](../ket_qua/20260119_134631/congestion_overview_combined.png) | Tổng quan tình trạng tắc nghẽn |
| ![MFD Scatter Combined](../ket_qua/20260119_134631/mfd_scatter_combined.png) | Macroscopic Fundamental Diagram |
| ![Efficiency Speed](../ket_qua/20260119_134631/efficiency_speed_plot.png) | So sánh hiệu quả tốc độ |
| ![Efficiency Volume](../ket_qua/20260119_134631/efficiency_volume_plot.png) | So sánh hiệu quả lưu lượng |
| ![Efficiency Occupancy](../ket_qua/20260119_134631/efficiency_occupancy_plot.png) | So sánh hiệu quả mật độ chiếm đường |


#### 💡 Nhận xét & Kết luận
- Điểm mạnh:
  - Lưu lượng tăng 20.48%
  - Trong toàn bộ thời gian mô phỏng, mạng lưới không xảy ra tình trạng tắc nghẽn (theo tiêu chí đánh giá)
- Điểm yếu/Vấn đề:
  - Độ chiếm dụng trung bình tăng 21.68%
  - Thuật toán chưa hội tụ, bài test này chỉ là thử đánh giá.
- Kết luận:
  - Chưa thể kết luận mạng lưới có cải thiện hay chưa
  - Vấn đề là trong các hàm phần thưởng sử dụng, bao gồm cả: hàm phần thưởng liên quan đến lưu lượng và hàm phần thưởng liên quan đến độ chiếm dụng. Nhưng trong bài thử nghiệm này chỉ cải thiện lưu lượng.
- Hướng cải tiến tiếp theo:
  - Tăng nhu cầu giao thông và đánh giá lại.
  - Training đến khi hội tụ (có thê hơi lâu)

---

### Experiment #003 - 2026-01-20
**Mục tiêu:** Kiểm tra đánh giá trên checkpoint mới (checkpoint_000018)

#### 🔧 Tham số (Parameters)
| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `learning_rate` | | |
| `batch_size` | | |
| `gamma` | | |
| `num_episodes` | 1 | Đánh giá 1 episode (theo terminal history) |
| | | |

#### 📈 Kết quả (Results)
| Metric | Giá trị | So sánh với baseline |
|--------|---------|---------------------|
| Mean Reward |~ -682 | |
| Episode Length |89 | |
| Convergence Step |16h | |
| Training Time | | |

#### 📉 Biểu đồ (nếu có)

##### So sánh tổng hợp (Before vs After)
| Biểu đồ | Mô tả |
|---------|-------|
| ![Congestion Overview Combined](../ket_qua/20260120_085246/congestion_overview_combined.png) | Tổng quan tình trạng tắc nghẽn |
| ![MFD Scatter Combined](../ket_qua/20260120_085246/mfd_scatter_combined.png) | Macroscopic Fundamental Diagram |
| ![Efficiency Speed](../ket_qua/20260120_085246/efficiency_speed_plot.png) | So sánh hiệu quả tốc độ |
| ![Efficiency Volume](../ket_qua/20260120_085246/efficiency_volume_plot.png) | So sánh hiệu quả lưu lượng |
| ![Efficiency Occupancy](../ket_qua/20260120_085246/efficiency_occupancy_plot.png) | So sánh hiệu quả mật độ chiếm đường |

#### 💡 Nhận xét & Kết luận
- Điểm mạnh:
  - Tổng lưu lượng tăng 69.3%
  - 
- Điểm yếu/Vấn đề:
  - Tuy lưu lượng tăng lớn, nhưng mạng lưới xuất hiện tình trạng tắc nghẽn
  - Độ chiếm dụng trung bình tăng 78%
- Kết luận:
  - Có thể kịch cách cấu hình mô phỏng khi chạy baseline và khi chạy thuật toán đang khác nhau.
- Hướng cải tiến tiếp theo:
  - Sửa lại cấu hình mô phỏng cho đồng nhât giữa chạy baselline và chạy thuật toán.

---

### Experiment #004 - 2026-01-22
**Mục tiêu:** Đánh giá hiệu quả thuật toán trên checkpoint mới, so sánh kết quả trước và sau với dữ liệu trong folder ket_qua/20260122_115608

#### 🔧 Tham số (Parameters)
| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `learning_rate` |0.0003  | |
| `batch_size` |  | |
| `gamma` |0.99  | |
| `num_episodes` | 1 | Đánh giá 1 episode |

#### 📈 Kết quả (Results)
| Metric | Giá trị | So sánh với baseline |
|--------|---------|---------------------|
| Mean Reward |~ -889| |
| Episode Length |8000s | |
| Convergence Step |  | |
| Training Time |33h35  | |

#### 📉 Biểu đồ (nếu có)

##### So sánh tổng hợp (Before vs After)
| Biểu đồ | Mô tả |
|---------|-------|
| ![Congestion Overview Combined](../ket_qua/20260122_115608/congestion_overview_combinedcombined.png) | Tổng quan tình trạng tắc nghẽn |
| ![MFD Scatter Combined](../ket_qua/20260122_115608/mfd_scatter_combined.png) | Macroscopic Fundamental Diagram |
| ![Efficiency Speed](../ket_qua/20260122_115608/efficiency_speed_plot.png) | So sánh hiệu quả tốc độ |
| ![Efficiency Volume](../ket_qua/20260122_115608/efficiency_volume_plot.png) | So sánh hiệu quả lưu lượng |
| ![Efficiency Occupancy](../ket_qua/20260122_115608/efficiency_occupancy_plot.png) | So sánh hiệu quả mật độ chiếm đường |

#### 💡 Nhận xét & Kết luận
- Điểm mạnh:
  - Tổng lưu lượng tăng 31.59%
- Điểm yếu/Vấn đề:
  - Độ chiếm dụng trung bình tăng 39,91%
  - Sau khi áp dụng thuât toán, mạng lưới tắc nghẽn hơn, mặc dù lưu lượng tăng nhiều.
- Kết luận:
  - Vấn đề có lẽ nằm ở chỗ hàm phần thưởng. Hiện tại thuật toán đang rất ưu tiên tăng lưu lượng nhưng không quan tâm tới các yếu tố khác
  - Một vấn đề nữa là mean total reward đang khác nhau giữa các lần chạy đánh giá, mặc dù kịch bản, mạng lưới, và các thông số mô phỏng giống hệt nhau. (liệu có phải do seed?)
- Hướng cải tiến tiếp theo:
  - Xem và sửa lại các hàm phần thưởng sao cho chuẩn.

<!--
### Experiment #XXX - YYYY-MM-DD
**Mục tiêu:** 

#### 🔧 Tham số (Parameters)
| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `learning_rate` | | |
| `batch_size` | | |
| `gamma` | | |
| `num_episodes` | | |
| | | |

#### 📈 Kết quả (Results)
| Metric | Giá trị | So sánh với baseline |
|--------|---------|---------------------|
| Mean Reward | | |
| Episode Length | | |
| Convergence Step | | |
| Training Time | | |

#### 📉 Biểu đồ (nếu có)


#### 💡 Nhận xét & Kết luận
- Điểm mạnh:
  - 
- Điểm yếu/Vấn đề:
  - 
- Kết luận:
  - 
- Hướng cải tiến tiếp theo:
  - 

---
-->

---

## 🔖 Bảng So Sánh Nhanh (Quick Comparison)

| Experiment | Date | Key Params | Mean Reward | Best? | Notes |
|------------|------|------------|-------------|-------|-------|
| #002 | 2026-01-19 | lr=0.0003, bs=1424 | ~ -145 | ⭐ | Baseline |
| #003 | 2026-01-20 | episodes=90 |~ -682 | | New Checkpoint, nhu cầu giao thông không lớn|
| #004 | 2026-01-22 | episodes=20 |~ -889 | | New Checkpoint, Nhu cầu giao thông lớn |
| | | | | | |

## 📝 Bảng Tracking Version Code

| Version | Date | Main Changes | Scope | Status |
|---------|------|------------|-------|--------|
| v1.0.0 | 2026-01-17 | Phiên bản khung dự án | Foundation | ✅ |
| v1.1.0 | 2026-01-18 | Episode-based training config | Configuration | ✅ |
| v1.1.1 | 2026-01-18 | Fix cấu hình đồng nhất | Config fix | ✅ |
| v1.1.2 | 2026-01-23 | Log(std) bounds + GraphSAGE review | Model | ✅ |
| v1.2.0 | 2026-01-23 | **Directional Adjacency Matrix** | **Major** | ✅ |
| v1.2.1 | 2026-01-23 | Code cleanup & Docstrings | Quality | ✅ |
| v1.2.2 | 2026-01-27 | **Fix Observation Structure (Lane-major)** | **Critical Fix** | ✅ **NEW** |

---

## 📒 Ghi Chú Chung

### Lessons Learned
- Reward_mean có thể khác nhau lớn giữa các lần training do kịch bản nhu cầu giao thông khác nhau.
- **[v1.2.0]** Lỗi logic vật lý trong GraphSAGE: Trước đây sử dụng một ma trận kề duy nhất cho tất cả hướng dẫn đến nhầm lẫn thông tin spatial. Bây giờ sử dụng ma trận riêng cho từng hướng, chính xác hơn về vật lý.
- Khi thiết kế GNN cho mô phỏng giao thông, cần phân biệt rõ hướng (direction) của neighbor để mô hình có thể học được pattern spatial phức tạp.

### TODO / Ideas
- [ ] **Training tiếp theo**: Huấn luyện mô hình với directional adjacency mới để kiểm tra hiệu quả cải thiện
- [ ] **Benchmark**: So sánh kết quả training v1.1.x (non-directional) vs v1.2.0 (directional) trên cùng kịch bản
- [ ] **Ablation Study**: Tắt directional adjacency để kiểm tra tác động thực tế đến hiệu quả
- [ ] **Mở rộng**: Xem xét thêm thông tin edge type (vd: highway vs local road) vào adjacency matrix
- [ ] **Optimization**: Kiểm tra xem directional adjacency có tăng thêm chi phí tính toán hay không

### Tài liệu tham khảo
- Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
- SUMO Network File Format: https://sumo.dlr.de/docs/Networks/index.html 

---

> **Hướng dẫn sử dụng:**
> 1. **Changelog:** Mỗi khi thay đổi code, copy template trong comment và điền thông tin
> 2. **Experiment:** Mỗi lần thử nghiệm tham số mới, copy template experiment và ghi kết quả
> 3. **Quick Comparison:** Cập nhật bảng so sánh nhanh để dễ nhìn tổng quan
> 4. Đánh số version theo format: `vMajor.Minor.Patch` (ví dụ: v1.0.0, v1.1.0, v2.0.0)
> 5. Đánh số experiment theo thứ tự: #001, #002, ...
