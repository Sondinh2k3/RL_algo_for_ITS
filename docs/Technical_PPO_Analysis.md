# Các Biến Thể Chính của PPO

Thuật toán PPO có hai biến thể phổ biến nhất:

### 1. PPO-Clip (Clipped Surrogate Objective)
- Đây là phiên bản được sử dụng trong dự án này và cũng là mặc định của RLlib, Stable-Baselines3, CleanRL...
- Sử dụng hàm loss với cơ chế "clipping" để giới hạn mức độ thay đổi của policy trong mỗi lần cập nhật.
- Ưu điểm: Đơn giản, hiệu quả, dễ tune hyperparameter.
- Công thức loss đã trình bày ở phần trên.

### 2. PPO-Penalty (Adaptive KL Penalty)
- Thay vì clipping, PPO-Penalty thêm một thành phần penalty vào loss dựa trên độ lệch KL-divergence giữa policy mới và cũ:
  $$L^{Penalty}(\theta) = \mathbb{E}_t [r_t(\theta) \hat{A}_t - \beta \cdot KL[\pi_{old}, \pi_\theta]]$$
- $\beta$ là hệ số penalty, có thể được điều chỉnh động dựa trên mức độ KL-divergence thực tế.
- Ưu điểm: Kiểm soát chính xác hơn mức độ thay đổi của policy, nhưng khó tune và dễ bị "quá phạt" (policy không học được gì nếu $\beta$ quá lớn).

### 3. So sánh nhanh
| Biến thể | Cơ chế ổn định | Dễ tune | Được dùng phổ biến |
|----------|---------------|---------|--------------------|
| PPO-Clip | Clipping      | Dễ      | ⭐⭐⭐⭐⭐             |
| PPO-Penalty | KL Penalty  | Khó     | ⭐                 |

**Kết luận:**
- PPO-Clip là lựa chọn mặc định cho hầu hết các framework RL hiện đại.
- PPO-Penalty chỉ dùng khi cần kiểm soát cực kỳ chặt chẽ về độ thay đổi policy, nhưng thường không cần thiết cho các bài toán thực tế.

---

# Phân Tích Kỹ Thuật: Proximal Policy Optimization (PPO) trong MGMQ

Tài liệu này mô tả chi tiết về thuật toán **Proximal Policy Optimization (PPO)** được áp dụng trong dự án điều khiển đèn tín hiệu giao thông (ITS) sử dụng kiến trúc MGMQ (Multi-Graph Masking Q-Network approach adapted for Policy Gradient).

## 1. Tổng Quan
PPO là một thuật toán *On-policy Gradient* tìm cách cân bằng giữa:
1.  **Ease of implementation:** Dễ cài đặt hơn TRPO.
2.  **Sample efficiency:** Tận dụng dữ liệu tốt hơn.
3.  **Ease of tuning:** Ít hyperparameter nhạy cảm hơn so với các thuật toán khác.

Trong dự án này, PPO được sử dụng để tối ưu hóa policy $\pi_\theta(a_t|s_t)$ nhằm điều khiển pha đèn giao thông cho 16 ngã tư (Agents) trong môi trường Grid 4x4.

---

## 2. PPO Objective Function (Hàm Mục Tiêu)

Hàm loss tổng quát của PPO trong project được tính như sau:

$$L_t(\theta) = \hat{\mathbb{E}}_t [ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) ]$$

Trong đó:
*   $L_t^{CLIP}$: Policy Loss (tối ưu hành động).
*   $L_t^{VF}$: Value Function Loss (tối ưu dự đoán phần thưởng).
*   $S$: Entropy Bonus (khuyến khích khám phá).
*   $c_1, c_2$: Các hệ số trọng số (`vf_loss_coeff`, `entropy_coeff`).

### 2.1. Clipped Surrogate Objective ($L^{CLIP}$)
Đây là thành phần cốt lõi giúp PPO hoạt động ổn định, ngăn chặn việc cập nhật policy quá mạnh làm "hỏng" những gì model đã học.

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

*   **Ratio $r_t(\theta)$**: Tỷ lệ xác suất hành động mới so với cũ.
    $$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
*   **Advantage $\hat{A}_t$**: Lợi thế của hành động $a_t$ so với mức trung bình. Được tính bằng **GAE (Generalized Advantage Estimation)**.
*   **Clipping $\epsilon$**: Giới hạn thay đổi. Trong config hiện tại: `clip_param: 0.2`.
    *   Nghĩa là policy mới không được lệch quá 20% so với policy cũ trong một bước update đơn lẻ.

### 2.2. Value Function Loss ($L^{VF}$)
Để tính Advantage, ta cần một Value Function $V(s)$ ước lượng tổng phần thưởng tích lũy.

$$L^{VF} = (V_\theta(s_t) - V_t^{target})^2$$

*   Trong `result.json`, `vf_loss` thường có giá trị rất lớn (ví dụ: ~5.0 - 6.0) do range của reward trong SUMO khá rộng (âm hàng trăm điểm).
*   Config hiện tại: `vf_clip_param: 10.0`. Điều này giúp cắt bớt các giá trị loss quá lớn do nhiễu, tránh gradient bùng nổ.

### 2.3. Entropy Bonus ($S$)
Entropy đo lường độ ngẫu nhiên của policy.
*   **Công thức:** $S = -\sum \pi(a|s) \log \pi(a|s)$
*   **Mục đích:** Ngăn policy hội tụ quá sớm vào một phương án cục bộ (sub-optimal).
*   **Vấn đề Log_std:** Với continuous action space (Gaussian distribution), entropy phụ thuộc vào `log_std`.
    *   Đã xử lý: Implement `LOG_STD_MAX = 2.0` trong `mgmq_model.py` để tránh entropy tăng vô hạn.
*   Config hiện tại: `entropy_coeff: 0.01`.

---

## 3. Cấu Hình Hyperparameters Thực Tế

Dựa trên file `result.json` mới nhất, đây là cấu hình đang chạy:

| Tham số | Giá trị | Ý nghĩa | Phân tích |
|---------|---------|---------|-----------|
| `gamma` | 0.99 | Discount Factor | Ưu tiên phần thưởng dài hạn (trọng số 0.99 cho tương lai). Phù hợp cho giao thông vì hành động hiện tại ảnh hưởng lâu dài. |
| `lambda` | 0.95 | GAE Parameter | Cân bằng giữa Bias và Variance khi tính Advantage. 0.95 là giá trị tiêu chuẩn. |
| `lr` | 0.0003 | Learning Rate | Tốc độ học. $3e-4$ là "hằng số vàng" của Adam Optimizer cho PPO. |
| `kl_target` | 0.01 | KL Divergence Target | Mức độ thay đổi phân phối mong muốn giữa các lần update. |
| `sgd_minibatch_size` | 64 | Batch Size cho SGD | Kích thước mẫu dùng để tính gradient trong mỗi bước update nhỏ. |
| `num_sgd_iter` | 4 | Epochs per Iteration | Số lần model học đi học lại trên cùng một batch dữ liệu thu thập được. |
| `entropy_coeff` | 0.01 | Trọng số Entropy | Khá nhỏ, chỉ khuyến khích explore nhẹ. Nếu model bị kẹt, có thể tăng lên 0.05. |

---

## 4. Kiến Trúc Model PPO-MGMQ

PPO trong dự án này không dùng mạng nơ-ron thẳng (MLP) thông thường mà sử dụng kiến trúc đồ thị (GNN) để trích xuất đặc trưng không gian:

1.  **Input:** Trạng thái giao thông (Queue length, wait time...) của 16 ngã tư.
2.  **Encoder (MGMQ):**
    *   **GAT (Graph Attention Network):** Học quan hệ giữa các làn đường trong một ngã tư.
    *   **GraphSAGE + BiGRU:** Học quan hệ giữa các ngã tư lân cận và thông tin chuỗi thời gian.
3.  **Policy Head (Actor):**
    *   Input: Joint Embedding từ Encoder.
    *   Output: Gaussian Distribution (Mean $\mu$, Log_std $\sigma$) cho mỗi phase đèn.
    *   **Lưu ý:** Output `log_std` đã được kẹp (clamp) trong khoảng $[-20, 2]$.
4.  **Value Head (Critic):**
    *   Input: Joint Embedding từ Encoder.
    *   Output: Một giá trị scalar $V(s)$ duy nhất.

---

## 5. Các Metrics Quan Trọng Cần Theo Dõi

Khi training, cần quan sát các chỉ số sau trong Tensorboard hoặc console (`result.json`):

1.  **`episode_reward_mean`**: Quan trọng nhất. Phải có xu hướng tăng (từ âm nhiều -> âm ít).
    *   Hiện tại: ~ -420. Đang dao động nhẹ, cần train lâu hơn.
2.  **`vf_explained_var`**: Giải thích phương sai của hàm giá trị.
    *   Tốt: > 0.5.
    *   Hiện tại: ~0.3 - 0.4. Cho thấy Critic (Value function) vẫn đang gặp khó khăn trong việc dự đoán chính xác reward, điều này bình thường ở giai đoạn đầu của môi trường phức tạp.
3.  **`entropy`**:
    *   Hiện tại: ~5.7. Khá cao. Nhờ cơ chế clamp `log_std`, nó sẽ không bùng nổ lên 9.0+ như trước.
    *   Kỳ vọng: Giảm dần từ từ theo thời gian khi Agent tự tin hơn vào hành động của mình.
4.  **`kl` (KL Divergence)**:
    *   Hiện tại: ~0.008 - 0.01. Rất tốt. Nó nằm gần `kl_target` (0.01), chứng tỏ Policy đang học ổn định, không thay đổi quá nhanh hay quá chậm.

---

## 6. Kết Luận

Setup PPO hiện tại đã được tinh chỉnh để phù hợp với bài toán Multi-Agent Traffic Control:
*   Đã khắc phục lỗi **Entropy Explosion** bằng việc giới hạn `log_std`.
*   Sử dụng **Episode-based batching** để phù hợp với độ dài mô phỏng thực tế.
*   Các siêu tham số (`gamma`, `lambda`, `clip_param`) đang ở mức tiêu chuẩn công nghiệp (Standard Baselines).

**Bước tiếp theo:** Tiếp tục train với số lượng Iteration lớn hơn (200-500) và theo dõi `episode_reward_mean`. Nếu reward không cải thiện, cần xem xét lại Reward Function trong file config.
