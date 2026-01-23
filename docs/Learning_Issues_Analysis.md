# 📊 Phân Tích Vấn Đề Learning Không Cải Thiện

**Ngày phân tích**: 2025-01-23  
**Tình trạng ban đầu**: episode_reward_mean ≈ -415 (không cải thiện sau 16 iterations)

---

## 🔍 Triệu Chứng Quan Sát Được

| Metric | Giá trị hiện tại | Giá trị mong đợi |
|--------|-----------------|------------------|
| `episode_reward_mean` | ~-415 (phẳng) | Giảm dần (cải thiện) |
| `policy_loss` | 0.005-0.017 | 0.01-0.1 |
| `vf_loss` | 5.0-6.0 | 0.1-1.0 |
| `vf_explained_var` | 0.17-0.55 | >0.7 |
| `entropy` | 5.7-6.2 | Giảm dần theo training |
| `policy/vf loss ratio` | 1:500 | ~1:10 |

---

## 🔴 Các Vấn Đề Nghiêm Trọng Được Phát Hiện

### 1. **Value Function Scaling Sai** ⭐ CRITICAL

**Vấn đề**: Thiếu `vf_loss_coeff` trong PPO config.

**Nguyên nhân**:
- Mặc định `vf_loss_coeff = 1.0` 
- `vf_loss = 5-6` chiếm ưu thế hoàn toàn trong total loss
- Policy gradients trở nên không đáng kể

**Hậu quả**:
- Critic được train quá mạnh, policy gần như không học
- `vf_explained_var` thấp (0.17-0.55) cho thấy critic vẫn chưa fit tốt

**Sửa chữa**:
```python
# train_mgmq_ppo.py
vf_loss_coeff=0.5  # Giảm từ 1.0 xuống 0.5
```

---

### 2. **Entropy Quá Cao - Policy Vẫn Random** ⭐ HIGH

**Vấn đề**: `LOG_STD_MAX = 2.0` cho phép std quá lớn.

**Phân tích**:
```
entropy = 0.5 * action_dim * (1 + log(2π) + 2*log_std)
Với LOG_STD_MAX = 2.0:
  std_max = e^2.0 ≈ 7.39
  entropy_max ≈ 0.5 * 4 * (1 + 1.84 + 4.0) = 13.68
```

Entropy 5.7-6.2 cho thấy policy vẫn quá random, không converge.

**Sửa chữa**:
```python
# mgmq_model.py
LOG_STD_MAX = 0.5  # Giảm từ 2.0 xuống 0.5
# std_max = e^0.5 ≈ 1.65 (hợp lý cho normalized actions)
```

---

### 3. **Reward Function Bug** ⭐ HIGH

**Vấn đề**: `_diff_departed_veh_reward()` có edge case bug.

**Code cũ (BUG)**:
```python
if initial > 0:
    ratio = departed / initial
else:
    if departed > 0:
        ratio = 1.0  # BUG: Cho max reward khi không có xe ban đầu!
```

**Vấn đề**:
- Khi `initial_vehicles = 0` và `departed > 0` → reward = 3.0 (maximum)
- Đây là tín hiệu sai lệch, không phản ánh đúng hiệu quả

**Sửa chữa**:
```python
MIN_VEHICLES_THRESHOLD = 1.0

if initial >= MIN_VEHICLES_THRESHOLD:
    ratio = departed / initial
else:
    if departed >= MIN_VEHICLES_THRESHOLD:
        ratio = 0.5  # Neutral-positive thay vì max
    else:
        return 0.0  # Không có xe → neutral
```

---

### 4. **Batch Size Quá Nhỏ** ⭐ MEDIUM

**Vấn đề**: `train_batch_size = 1424` với 16 agents.

**Phân tích**:
```
samples_per_agent = 1424 / 16 = 89 samples
→ Variance cao trong gradient estimates
```

**Sửa chữa**:
```python
train_batch_size=4096,  # Tăng từ 1424
minibatch_size=128,     # Tăng từ 64
num_epochs=10,          # Tăng từ 4
```

---

## ✅ Tổng Hợp Các Thay Đổi

### File: `scripts/train_mgmq_ppo.py`

| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `vf_loss_coeff` | (mặc định 1.0) | 0.5 | Cân bằng policy/vf loss |
| `train_batch_size` | 1424 | 4096 | Giảm gradient variance |
| `minibatch_size` | 64 | 128 | Better batch normalization |
| `num_epochs` | 4 | 10 | Thorough updates |

### File: `src/models/mgmq_model.py`

| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `LOG_STD_MAX` | 2.0 | 0.5 | Entropy converge nhanh hơn |

### File: `src/environment/drl_algo/traffic_signal.py`

| Function | Thay đổi |
|----------|----------|
| `_diff_departed_veh_reward()` | Fix edge case khi initial_vehicles ≈ 0 |

---

## 📈 Kỳ Vọng Sau Khi Fix

1. **Policy loss** tăng lên ~0.01-0.05 (có gradient đủ lớn để learn)
2. **VF loss** giảm dần về ~0.5-1.0 khi critic converge
3. **VF explained variance** tăng lên >0.7
4. **Entropy** giảm dần khi policy converge
5. **Episode reward** cải thiện (tăng dần từ -415)

---

## 🧪 Khuyến Nghị Test

1. **Train ít nhất 100-200 iterations** để thấy trend
2. **Monitor các metrics sau**:
   - `episode_reward_mean`: Phải có xu hướng tăng
   - `vf_loss`: Phải giảm dần
   - `policy_loss`: Phải ổn định ~0.01-0.05
   - `entropy`: Phải giảm dần
   - `kl_divergence`: Phải < `kl_target` (0.01)

3. **Nếu vẫn không improve sau 200 iterations**:
   - Thử dùng single reward function: `--reward-fn queue`
   - Giảm learning rate xuống 1e-4
   - Kiểm tra observation normalization

---

## 📚 Tham Khảo

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [RLlib PPO Documentation](https://docs.ray.io/en/latest/rllib/algorithms.html#ppo)
