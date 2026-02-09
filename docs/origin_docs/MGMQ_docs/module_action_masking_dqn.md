# Mô-đun MGMQ: Double DQN & Cơ chế Action Masking

Dựa trên tài liệu: *"A large-scale traffic signal control algorithm based on multi-layer graph deep reinforcement learning.pdf"* (Tao Wang et al.)

MGMQ sử dụng thuật toán **Double DQN (DDQN)** làm lõi ra quyết định, kết hợp với một cơ chế gán mặt nạ hành động (Action Masking) để xử lý các ràng buộc vật lý tại các nút giao khác nhau.

## 1. Thuật toán Double DQN (DDQN)
Để tránh hiện tượng ước lượng quá mức (overestimation) giá trị Q thường gặp trong DQN truyền thống, MGMQ sử dụng hai mạng thần kinh:
- **Current Q-network ($\theta$)**: Dùng để chọn hành động có giá trị Q cao nhất.
- **Older Q-network ($\theta'$)**: Dùng để tính toán giá trị Q mục tiêu định kỳ.

Hành động $a^*$ được chọn dựa trên chính sách $\epsilon$-greedy:
- Với xác suất $1-\epsilon$: Chọn hành động tối ưu $a = \text{argmax}_a Q(s, a)$.
- Với xác suất $\epsilon$: Chọn hành động ngẫu nhiên để khám phá môi trường.

## 2. Cơ chế Action Masking (Gán mặt nạ)
Đây là kỹ thuật cốt lõi giúp MGMQ có thể áp dụng cho bất kỳ cấu trúc nút giao nào (ví dụ: nút giao chữ T có 3 pha, nút giao chữ thập có 4 pha) mà không cần thay đổi kích thước mạng thần kinh.

- **Vấn đề**: Kích thước đầu ra của mạng Q được cố định theo số pha lớn nhất (thường là 4). Tại các nút giao có ít pha hơn, các hành động dư thừa là không hợp lệ.
- **Giải pháp**:
    - Đối với mỗi nút giao $j$, mô hình duy trì một danh sách các pha đèn hợp lệ $A_j$.
    - Trước khi thực hiện `argmax`, giá trị Q của các hành động không nằm trong $A_j$ sẽ bị ghi đè bằng giá trị âm vô cùng ($-\infty$):
    $$Q(s, a; \theta) = -\infty, \quad \text{nếu } a \notin A_j$$
- **Kết quả**: Tác nhân sẽ bị "ép" chỉ được chọn các pha đèn khả thi về mặt vật lý tại nút giao đó.

## 3. Khả năng mở rộng (Scalability)
- **Chia sẻ tham số (Parameter Sharing)**: Tất cả các tác nhân trên toàn mạng lưới sử dụng chung một bộ trọng số $\theta$. Điều này làm giảm đáng kể số lượng tham số cần huấn luyện và cho phép mô hình học được các đặc trưng chung của luồng giao thông.
- **DQL Framework**: Việc sử dụng phương pháp dựa trên giá trị (Value-based) thay vì chính sách (Policy-based) giúp MGMQ ổn định hơn khi đối mặt với không gian trạng thái khổng lồ của mạng lưới giao thông đô thị.
- **Zero-training Transfer**: Nhờ vào Action Masking, một mô hình được huấn luyện trên mạng lưới giả lập đơn giản có thể được triển khai trực tiếp lên các nút giao thực tế có hình học phức tạp mà không cần cấu hình lại.
