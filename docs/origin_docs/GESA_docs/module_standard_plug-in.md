# Mô-đun GPI (General Plug-In) trong GESA

Dựa trên tài liệu: *2024 - A general scenario-agnostic reinforcement learning for traffic signal control.pdf*

Mô-đun GPI là thành phần then chốt giúp GESA độc lập với kịch bản (scenario-agnostic), cho phép tự động xử lý mọi cấu trúc nút giao mà không cần gán nhãn thủ công.

## 1. Chuẩn hóa Hướng tiếp cận (Unifying Approaches)
Mục tiêu là căn chỉnh các hướng của nút giao mục tiêu với các hướng của nút giao chuẩn (North, South, East, West).

### Thuật toán 1: Chuẩn hóa hướng tiếp cận (To unify the approach directions)
**Đầu vào**: Số lượng tiếp cận của nút giao mục tiêu $|a|$; Các vector vị trí của tiếp cận $V = \{v_1, v_2, ..., v_{|a|}\}$.

**Các bước thực hiện**:
1. **Khởi tạo**: Đặt `Flag = False` (chỉ thị thành công).
2. **Vòng lặp bên ngoài**: Thử lần lượt từng tiếp cận $a^*$ trong tập $\{a\}_s$ làm hướng Bắc tham chiếu.
3. **Vòng lặp bên trong**: Với mỗi tiếp cận $a \ne a^*$:
    a. Tính góc: $\theta_{a, a^*} = \arccos\left( \frac{v_a \cdot v_{a^*}}{\|v_a\| \|v_{a^*}\|} \right)$.
    b. Xác định hướng quay qua tích có hướng: Nếu $v_a \times v_{a^*} < 0$ thì $\theta_{a, a^*} = 360^\circ - \theta_{a, a^*}$.
    c. Phân loại tiếp cận vào bộ hướng $M = \{m_N, m_S, m_W, m_E\}$:
        - **East ($m_E$)**: $45^\circ \le \theta_{a, a^*} < 135^\circ$
        - **South ($m_S$)**: $135^\circ \le \theta_{a, a^*} < 225^\circ$
        - **West ($m_W$)**: $225^\circ \le \theta_{a, a^*} < 315^\circ$
        - **North ($m_N$)**: Các trường hợp còn lại.
4. **Kiểm tra**: Nếu sau khi phân loại, mỗi hướng trong $M$ chứa không quá 1 tiếp cận (`length < 2`), đặt `Flag = True` và kết thúc thuật toán.

## 2. Chuẩn hóa Chuyển động (Unifying Movements)
Mục tiêu là đạt được sự khớp nối 1-1 giữa làn đường và các chuyển động chuẩn, đồng thời xử lý các pha không khả dụng.

### Quy tắc ưu tiên chuyển động:
Đối với các làn đường hỗn hợp (cho phép nhiều chuyển động), GPI phân cấp ưu tiên dựa trên luật giao thông:
1. **Chuyển động Đi thẳng (Through)**: Ưu tiên cao nhất.
2. **Chuyển động Rẽ trái (Left)**: Ưu tiên thứ hai.
3. **Chuyển động Rẽ phải (Right)**: Thường được coi là tự do và không cần tín hiệu kiểm soát.

Kết quả là 8 pha tín hiệu giao thông được tạo ra đồng nhất cho mọi nút giao. Các tín hiệu đỏ được sử dụng để lấp đầy các pha không khả dụng (unavailable phases).

## 3. Vai trò Kỹ thuật
- **Tự động hóa**: Thay thế hoàn toàn việc gán nhãn thủ công cấu trúc làn đường cho hàng trăm nút giao.
- **Tính khởi tạo**: Mô-đun này được đặt ở giai đoạn khởi tạo (initialization) của kịch bản mô phỏng, đảm bảo không gây trễ trong quá trình chạy.
- **Tính tương thích**: Giúp một mô hình RL duy nhất có thể quan sát và điều khiển các nút giao 3 hướng, 4 hướng hoặc các nút giao có góc lệch không đều một cách nhất quán.
