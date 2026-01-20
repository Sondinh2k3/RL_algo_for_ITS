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

<!-- TEMPLATE CHO EXPERIMENT MỚI - Copy phần này khi thêm experiment mới -->
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
| #003 | 2026-01-20 | episodes=90 |~ -682 | | New Checkpoint |
| | | | | | |

---

## 📒 Ghi Chú Chung

### Lessons Learned
- Reward_mean có thể khác nhau lớn giữa các lần training do kịch bản nhu cầu giao thông khác nhau.

### TODO / Ideas
- [ ] 
- [ ] 

### Tài liệu tham khảo
- 

---

> **Hướng dẫn sử dụng:**
> 1. **Changelog:** Mỗi khi thay đổi code, copy template trong comment và điền thông tin
> 2. **Experiment:** Mỗi lần thử nghiệm tham số mới, copy template experiment và ghi kết quả
> 3. **Quick Comparison:** Cập nhật bảng so sánh nhanh để dễ nhìn tổng quan
> 4. Đánh số version theo format: `vMajor.Minor.Patch` (ví dụ: v1.0.0, v1.1.0, v2.0.0)
> 5. Đánh số experiment theo thứ tự: #001, #002, ...
