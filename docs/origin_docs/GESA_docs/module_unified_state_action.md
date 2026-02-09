# Không gian Trạng thái và Hành động Thống nhất trong GESA

Dựa trên tài liệu: *2024 - A general scenario-agnostic reinforcement learning for traffic signal control.pdf*

Để đảm bảo khả năng chuyển giao của mô hình, GESA thống nhất kích thước đầu vào và đầu ra cho mọi nút giao, bất kể độ phức tạp của kịch bản.

## 1. Không gian Trạng thái Thống nhất (General State Space)

GESA thu thập quan sát theo 8 nhóm chuyển động (movements) của nút giao chuẩn.

### Các đặc trưng của chuyển động (Movement Features):
Mỗi chuyển động được biểu diễn bởi 5 đặc trưng chính:
1. **Queue length**: Độ dài hàng đợi xe.
2. **Current phase**: Trạng thái pha hiện tại (dạng nhị phân).
3. **Occupancy**: Độ chiếm dụng làn đường.
4. **Flow**: Lưu lượng giao thông.
5. **Number of stopping cars**: Số lượng xe đang dừng.

### Cơ chế xử lý đặc thù:
- **Tổng hợp (Aggregation)**: Nếu một chuyển động có nhiều làn đường vào, đặc trưng của các làn sẽ được lấy trung bình để đại diện cho chuyển động đó.
- **Lấp đầy (Zero Padding)**: Nếu một chuyển động bị thiếu (ví dụ nút giao chữ T), đặc trưng của nó sẽ được lấp đầy bằng giá trị 0.
- **Chỉ thị nhị phân (Binary Indicator)**: Một đặc trưng nhị phân bổ sung được thêm vào để báo hiệu cho mô hình biết chuyển động đó có tồn tại thực sự hay không.

## 2. Không gian Hành động Thống nhất (General Action Space)

Hành động của tác nhân là chọn pha cho khoảng thời gian tiếp theo.

### Cấu trúc pha chuẩn:
- GESA bảo lưu **8 pha đèn chuẩn** với thứ tự cố định.
- Hành động là vector $a_i \in \mathbb{R}^8$ tương ứng với 8 pha.

### Cơ chế Masking (Action Mask):
Vì các nút giao thực tế có thể không có đủ 8 pha:
- **Nhận diện**: Sử dụng cấu trúc nút giao được ánh xạ bởi GPI để xác định các pha không hợp lệ.
- **Mặt nạ (Mask)**: Thêm mặt nạ để tác nhân bỏ qua (ignore) các pha không có sẵn này trong quá trình ra quyết định.

## 3. Lợi ích kỹ thuật
- **Constant Dimensions**: Kích thước in-out của mô hình luôn không đổi cho phép một bộ tham số mạng thần kinh duy nhất điều khiển toàn bộ mạng lưới giao thông.
- **Transferability**: Tác nhân có thể học được logic điều khiển chung (chẳng hạn: ưu tiên hướng có hàng đợi dài nhất) thay vì học cách điều khiển từng index làn đường cụ thể của một kịch bản nhất định.
