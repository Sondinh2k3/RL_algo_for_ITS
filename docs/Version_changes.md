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
| #001 | 2026-01-17 | lr=0.001, bs=64 | 150.5 | ⭐ | Baseline |
| | | | | | |

---

## 📒 Ghi Chú Chung

### Lessons Learned
- 

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
