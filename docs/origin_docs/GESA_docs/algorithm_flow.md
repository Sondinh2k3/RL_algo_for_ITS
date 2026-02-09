# Luồng Thuật toán GESA

Dựa trên tài liệu: *2024 - A general scenario-agnostic reinforcement learning for traffic signal control.pdf*

Thuật toán GESA vận hành qua ba giai đoạn xử lý dữ liệu chính để đưa ra quyết định điều khiển đèn tín hiệu:

## 1. Giai đoạn Tổng quát hóa (Unification Stage - GPI)
Mô-đun GPI (General Plug-In) thực hiện ánh xạ nút giao bất kỳ vào cấu trúc chuẩn:
- **Chuẩn hóa hướng (Approach Unification)**: Căn chỉnh các hướng tiếp cận thực tế vào các hướng North (N), South (S), East (E), West (W).
- **Chuẩn hóa chuyển động (Movement Unification)**: Thiết lập ánh xạ 1-1 giữa các làn đường và 8 chuyển động giao thông chuẩn dựa trên thứ tự ưu tiên (Đi thẳng > Rẽ trái > Rẽ phải).
- **Xử lý thiếu hụt**: Sử dụng zero-padding cho các chuyển động bị thiếu và thêm vector chỉ thị (binary indicator) để đánh dấu sự tồn tại của chuyển động.

## 2. Giai đoạn Nhúng đặc trưng (Feature Embedding Stage)
- **Nhúng chuyển động (Movement Embedding)**: 
    - Thu thập 5 đặc trưng (Queue, Phase, Occupancy, Flow, Stopping cars) cho mỗi chuyển động.
    - Áp dụng MLP và Concatenation cho từng đặc trưng của chuyển động thứ $j$.
    - Công thức: $f̄m_{i,j} = \parallel_{n=1}^{N+1} MLP(\tilde{f}m_{i,j,n})$
- **Nhúng pha (Phase Embedding)**: Cộng các nhúng chuyển động tương ứng để tạo đại diện pha: $fp_{i,l} = f̄m_{i,j} + f̄m_{i,j'}$.
- **Đại diện cặp pha (Phase-pair representation)**: Xây dựng để nắm bắt quan hệ đối kháng giữa các pha: $fpp_i = fp_{i,l} \parallel fp_{i,l'}$.

## 3. Giai đoạn Ra quyết định (Decision Stage - Actor-Critic)
- **Cạnh tranh pha (Phase Competition)**: Sử dụng mặt nạ cạnh tranh (Competition Mask $\Omega$) và tích Kronecker ($\otimes$) để xử lý xung đột: $fpp_{masked,i} = Conv_{1\times1}(fpp_i) \otimes \Omega$.
- **Mạng Actor**: 
    - Tính toán xác suất chọn pha dựa trên tổng các nhúng cặp pha đã qua mặt nạ.
    - Công thức: $\pi(a_i|f_i) = \text{Softmax}(MLP(\sum(fpp_{actor,i})))$.
- **Mạng Critic**: 
    - Ước lượng giá trị trạng thái $V(f_i)$ để hỗ trợ cập nhật chính sách.
    - Công thức: $V(f_i) = MLP(\sum(fpp_{critic,i}))$.

## 4. Chu kỳ điều khiển
Tại mỗi bước thời gian $t$, Agent sẽ:
1. Quan sát trạng thái giao thông.
2. Xử lý qua mô-đun GPI và FRAP cải tiến.
3. Chọn pha đèn có xác suất cao nhất hoặc lấy mẫu từ phân phối để khám phá.
4. Thực thi hành động và nhận phần thưởng $R_t$ dựa trên tổng trọng số của Queue, Wait time, Delay và Pressure.
