# Các vấn đè cần xem xét:


### 1. Vấn đề về việc sử dụng các API cũ của RLlib:

Trong file [scripts/train_mgmq_ppo.py](../scripts/train_mgmq_ppo.py), tại hàm [`create_mgmq_ppo_config()`](../scripts/train_mgmq_ppo.py), ngay phía dưới chỗ sau khi gọi [`PPOConfig()`](../scripts/train_mgmq_ppo.py), hiện tại đang phải dùng API cũ vì API RLlib mới đang không tương thích với custom model MGMQ tự định nghĩa. 

- **Vấn đề:** Hiện tại custom model đang kế thừa từ [`TorchModelV2`](https://docs.ray.io/en/latest/rllib/rllib-models.html#torchmodelv2), để dùng API mới, cần viết lại custom model dưới dạng [`RLModule`](https://docs.ray.io/en/latest/rllib/rllib-models.html#custom-rlmodules).