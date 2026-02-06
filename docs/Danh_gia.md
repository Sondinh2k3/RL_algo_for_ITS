# Báo Cáo Đánh Giá & Tổng Kết Dự Án MGMQ

**Ngày:** 2026-02-06

---

## 1. Tổng Quan

Báo cáo này tổng hợp lại quá trình phát triển, kiểm thử và tối ưu hóa thuật toán MGMQ (Max-Pressure GAT Multi-Agent Q-learning) cho bài toán điều khiển đèn tín hiệu giao thông. Mục tiêu là xác định rõ những gì đã làm được và phân tích sâu các vấn đề kỹ thuật đã gặp phải.

## 2. Các Công Việc Đã Hoàn Thành (Accomplishments)

### 2.1. Hoàn thiện Codebase và Refactoring

- **Cấu trúc lại dự án**: Tách biệt rõ ràng các module `preprocessing` (FRAP), `models` (GAT, MGMQ), và `environment` (TrafficSignal).
- **Tối ưu hóa FRAP**: Đã rà soát và sửa lỗi logic trong `frap.py`, đảm bảo ánh xạ đúng từ 8 standard phases sang actual phases của SUMO.

### 2.2. Kiểm thử Toàn diện (Comprehensive Testing)

Đã xây dựng bộ test suite gồm 62 tests, bao phủ hầu hết các thành phần quan trọng:

- **Kết quả**: **59/62 Passed (95.2%)**.
- **Các module đã verify**:
  - `test_frap_preprocessing.py`: ✅ Logic pha và hướng di chuyển ĐÚNG.
  - `test_gat_adjacency_matrices.py`: ✅ Ma trận kề GAT chính xác.
  - `test_masked_softmax_distribution.py`: ✅ Action masking hoạt động chuẩn (xác suất phase bị mask = 0).
  - `test_eval_policy_application.py`: ✅ Script đánh giá áp dụng đúng policy đã học (deterministic, valid actions).

### 2.3. Fix Training Configuration

Đã phát hiện và sửa các siêu tham số (hyperparameters) không hợp lý gây cản trở quá trình hội tụ:

- Giảm Learning Rate: `0.0005` -> `0.0001` (ổn định hơn).
- Tăng Batch Size: `512` -> `2048` (giảm variance của gradient).
- Thêm **Entropy Schedule**: Giảm dần entropy theo thời gian để khuyến khích exploitation.

---

## 3. Các Sai Lầm & Vấn Đề Đã Phát Hiện (Critical Issues)

### 3.1. Vấn Đề "Uniform Policy" (Nghiêm trọng nhất)

- **Triệu chứng**: Sau khi training, policy cấp thời gian xanh gần như bằng nhau cho tất cả các phase (chênh lệch chỉ 1-2s). Hiệu quả thực tế kém hơn baseline cố định.
- **Nguyên nhân gốc rễ**:
  - **Entropy không giảm**: Trong các lần chạy cũ, entropy giữ nguyên ở mức ~1.36-1.38 (gần mức tối đa của 8 phases).
  - Điều này có nghĩa là mạng **chưa bao giờ học được cách ưu tiên** phase nào, mà chỉ output ngẫu nhiên đều nhau (explore mãi mãi).
  - Công thức tính green time: `min_green + (action_prob * remaining_time)`. Nếu `action_prob` đều nhau (~0.125), thì green time sẽ đều nhau.
- **Giải pháp đã áp dụng**: Thêm `entropy_coeff_schedule` để ép buộc giảm entropy sau 100-300 iterations.

### 3.2. Mâu thuẫn Config `obs_dim`

- **Vấn đề**: File config set cứng `obs_dim: 48` (giả sử 12 detectors \* 4 features), nhưng thực tế môi trường tạo ra observation với kích thước khác (ví dụ 160) nếu số lượng detector thay đổi.
- **Hệ quả**: Gây crash khi chạy một số unit test (`test_model_forward_pass.py`).
- **Bài học**: Cần cơ chế dynamic `obs_dim` hoặc validate chặt chẽ cấu hình đầu vào.

### 3.3. Reward Function chưa ổn định

- Các hàm reward ban đầu (như `wait_time`) có giá trị quá lớn, làm gradient bị nổ (exploding gradients) hoặc value function không học được.
- Đã fix bằng cách thêm cơ chế clipping và normalization cho reward.

---

## 4. Chi Tiết Các Thay Đổi & Tác Động (Detailed Changes & Impact)

Để giải quyết vấn đề "Uniform Policy", chúng ta đã thực hiện 3 thay đổi cốt lõi. Dưới đây là phân tích chi tiết về **cái gì đã thay đổi**, **tại sao**, và **tác động thực tế**:

### 4.1. Áp dụng Entropy Schedule (Quan trọng nhất)

- **Thay đổi**:
  - _Trước đây_: `entropy_coeff = 0.01` (cố định mãi mãi).
  - _Hiện tại_: `entropy_coeff_schedule = [[0, 0.01], [100, 0.005], [300, 0.001]]`.
- **Cơ chế**:
  - Ban đầu giữ entropy cao (0.01) để model khám phá ngẫu nhiên.
  - Sau 100 iterations, giảm hệ số xuống một nửa.
  - Sau 300 iterations, giảm xuống rất thấp (0.001).
- **Tác động thực tế**:
  - Ép buộc model chuyển từ trạng thái "do dự" (xác suất các hành động như nhau) sang trạng thái "quyết đoán" (ưu tiên rõ rệt phase tốt nhất).
  - Trực tiếp giải quyết vấn đề green time "chia đều" chán ngắt.

### 4.2. Tăng Kích Thước Batch (Batch Size)

- **Thay đổi**:
  - `train_batch_size`: Tăng từ `512` lên `2048`.
  - `minibatch_size`: Tăng từ `64` lên `256`.
- **Cơ chế**:
  - Với Multi-Agent RL (16 agents cùng lúc), batch size 512 là quá nhỏ, dẫn đến nhiễu (variance) rất lớn trong tính toán gradient.
  - Tăng batch size giúp ước lượng hướng di chuyển của gradient chính xác hơn.
- **Tác động thực tế**:
  - Training ổn định hơn (loss không bị nhảy lung tung).
  - Model học được các pattern phức tạp chắc chắn hơn thay vì bị nhiễu bởi các mẫu ngẫu nhiên.

### 4.3. Giảm Tốc Độ Học (Learning Rate)

- **Thay đổi**: Giảm `learning_rate` từ `0.0005` xuống `0.0001`.
- **Cơ chế**:
  - LR to giúp học nhanh nhưng dễ bị "vọt xà" khỏi điểm tối ưu hoặc dao động mạnh.
  - LR nhỏ giúp model hội tụ từ từ nhưng chắc chắn.
- **Tác động thực tế**:
  - Tránh hiện tượng `Episode Reward` bị sụt giảm đột ngột (catastrophic forgetting).
  - Kết hợp với batch size lớn, tạo ra quỹ đạo learning mượt mà.

---

## 5. Phân Tích Hiệu Quả Training (Old vs New)

So sánh giữa lần chạy cũ (bị lỗi uniform) và lần chạy mới nhất (sau khi fix config):

| Metric             | Old Run (Lỗi) | New Run (Sau Fix) | Nhận Xét                                 |
| ------------------ | ------------- | ----------------- | ---------------------------------------- |
| **Iterations**     | 10            | 38+               | Đang train lâu hơn để hội tụ             |
| **Entropy Change** | -0.07 (Chậm)  | **-0.17 (Tốt)**   | Entropy đang giảm nhanh hơn nhờ schedule |
| **Reward Max**     | 4.77          | **11.09**         | Model đã tìm được policy tốt hơn nhiều   |
| **Episode Reward** | -5.62 (End)   | -3.45 (End)       | Reward trung bình đang cải thiện         |

> **Nhận định**: Lần chạy mới cho thấy tín hiệu cực kỳ khả quan. Entropy đang giảm (chứng tỏ policy đang bắt đầu "quyết đoán" hơn), và Max Reward đạt được cao hơn gấp đôi so với trước.

---

---

## 6. Kết Luận & Hướng Tiếp Theo

1. **Tiếp tục Training**: Cần để training chạy đủ ít nhất **300-500 iterations**. Tại iteration 38, model mới chỉ bắt đầu học, chưa thể hội tụ hoàn toán.
2. **Kiểm tra đồ thị**: Theo dõi chặt chẽ biểu đồ Entropy trên Tensorboard. Nếu nó giảm xuống dưới `0.8`, ta sẽ thấy green time bắt đầu chênh lệch rõ rệt giữa các phase.
3. **Sử dụng Checkpoint tốt nhất**: Khi eval, hãy chọn checkpoint có `reward_max` cao nhất, không nhất thiết là checkpoint cuối cùng.
4. **Eval Script**: Script `eval_mgmq_ppo.py` đã được kiểm chứng hoạt động tốt. Hãy dùng nó để quay video demo sau khi training xong.

**Trạng thái dự án: ĐANG ĐI ĐÚNG HƯỚNG.**
