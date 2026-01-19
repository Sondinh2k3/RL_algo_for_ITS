# Đánh giá Yêu cầu Phần cứng cho Hệ thống MGMQ-PPO

Tài liệu này phân tích và đề xuất cấu hình phần cứng phù hợp để vận hành và huấn luyện mô hình MGMQ (Multi-Graph Multi-Agent Reinforcement Learning) với môi trường mô phỏng SUMO.

## 1. Tổng quan Tải công việc (Workload Overview)

Hệ thống bao gồm hai thành phần tiêu tốn tài nguyên chính:
1.  **Môi trường Mô phỏng (SUMO Simulation)**:
    - Chạy trên CPU.
    - Mỗi worker (tác nhân thu thập dữ liệu) chạy một instance SUMO độc lập.
    - Tốn nhiều CPU và RAM tùy thuộc vào kích thước mạng lưới giao thông và số lượng xe.
2.  **Huấn luyện Mô hình (Deep RL Training)**:
    - Chạy trên GPU (hoặc CPU nếu không có GPU).
    - Mô hình MGMQ sử dụng kiến trúc phức tạp: GNN (Graph Attention Network) kết hợp với RNN (Bi-GRU) để xử lý dữ liệu không gian - thời gian.
    - Yêu cầu khả năng tính toán ma trận lớn.

## 2. Phân tích Thực nghiệm (Experimental Analysis)

Dựa trên lần chạy thử nghiệm gần nhất với lệnh:
`python scripts/train_mgmq_ppo.py --network grid4x4 --gpu`

- **Tài nguyên sẵn có**: 16 CPUs, 1 GPU.
- **Cấu hình mặc định**: `num_workers=2`.
- **Sử dụng thực tế**:
    - **CPU**: ~3.0 / 16 cores (1 Driver + 2 Workers).
    - **GPU**: 1.0 / 1 (Sử dụng cho quá trình update gradient).
- **Hiệu suất**: Thời gian mỗi vòng lặp (iteration) khoảng 120 giây.

**Nhận xét**: Hệ thống hiện tại đang **chưa tận dụng hết tài nguyên CPU**. Với 16 cores, chúng ta chỉ mới sử dụng khoảng 20% công suất. Việc tăng số lượng worker sẽ giúp thu thập dữ liệu nhanh hơn đáng kể, từ đó giảm thời gian huấn luyện.

## 3. Đề xuất Cấu hình Phần cứng (Hardware Recommendations)

### A. CPU (Bộ vi xử lý) - Quan trọng nhất
CPU đóng vai trò quyết định tốc độ thu thập dữ liệu (rollout). Ray RLlib song song hóa việc này bằng cách chạy nhiều worker.

*   **Yêu cầu**: Đa nhân (Multi-core).
*   **Khuyến nghị**:
    *   **Tối thiểu**: 4-8 cores (cho các thử nghiệm nhỏ).
    *   **Khuyên dùng**: 16-32 cores trở lên.
    *   **Lý do**: Mỗi worker chiếm 1 core CPU. Để huấn luyện hiệu quả trên mạng lưới lớn (như `grid4x4` hoặc lớn hơn), bạn nên chạy 10-20 workers song song.

### B. RAM (Bộ nhớ trong)
RAM cần đủ để chứa các instance của SUMO và bộ đệm kinh nghiệm (experience buffer) của Ray.

*   **Yêu cầu**: Phụ thuộc vào số lượng worker.
*   **Ước tính**:
    *   Mỗi worker (SUMO + Python env) tiêu tốn khoảng 1GB - 2GB RAM (tùy độ phức tạp mạng lưới).
    *   Driver/Trainer tiêu tốn thêm 2-4GB.
*   **Khuyến nghị**:
    *   **Tối thiểu**: 16 GB.
    *   **Khuyên dùng**: 32 GB - 64 GB (để chạy full 16-32 workers mượt mà).

### C. GPU (Card đồ họa)
GPU chịu trách nhiệm tính toán gradient cho mô hình MGMQ.

*   **Yêu cầu**: Hỗ trợ CUDA (NVIDIA).
*   **Khuyến nghị**:
    *   **VRAM**: 8GB trở lên. Mô hình GNN + RNN với batch size lớn (ví dụ: 4096 hoặc 8192) sẽ tốn VRAM.
    *   **Dòng card**: NVIDIA RTX 3060/4060 trở lên là đủ tốt. Đối với training quy mô lớn, RTX 3090/4090 hoặc A-series sẽ tối ưu hơn.

### D. Lưu trữ (Storage)
*   **Khuyến nghị**: SSD (NVMe càng tốt).
*   **Lý do**: Ghi log (TensorBoard), checkpoint mô hình và tải dữ liệu bản đồ SUMO nhanh chóng.

## 4. Tối ưu hóa Cấu hình Chạy (Optimization Tips)

Để tận dụng tối đa phần cứng hiện tại (16 CPUs, 1 GPU), bạn nên điều chỉnh tham số trong `scripts/train_mgmq_ppo.py`:

1.  **Tăng số lượng Worker**:
    *   Đặt `num_workers` khoảng 14 hoặc 15 (để lại 1-2 core cho hệ điều hành và Driver).
    *   Ví dụ: `num_workers=14`.
    *   Điều này sẽ tăng tốc độ thu thập mẫu lên gấp ~7 lần so với hiện tại (2 workers).

2.  **Batch Size**:
    *   Khi tăng worker, hãy đảm bảo `train_batch_size` đủ lớn (ví dụ: `num_workers * rollout_fragment_length`).

3.  **GPU**:
    *   Tiếp tục sử dụng cờ `--gpu` để tận dụng GPU cho việc cập nhật trọng số mô hình.

## 5. Kết luận

Với cấu hình hiện tại (16 CPUs, 1 GPU), máy trạm của bạn **hoàn toàn đủ khả năng** để huấn luyện mô hình MGMQ cho mạng lưới `grid4x4`. Tuy nhiên, cần điều chỉnh tham số phần mềm (`num_workers`) để khai thác hết sức mạnh của CPU, giúp giảm thời gian huấn luyện đáng kể.
