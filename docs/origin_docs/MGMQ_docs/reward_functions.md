# MGMQ: Hàm thưởng Đa mục tiêu (Multi-objective Reward)

Dựa trên tài liệu: *"A large-scale traffic signal control algorithm based on multi-layer graph deep reinforcement learning.pdf"* (Tao Wang et al.)

Trong khung làm việc MGMQ, hàm thưởng được thiết kế để cân bằng giữa việc giải tỏa xe nhanh chóng và giảm thiểu sự ùn tắc hàng đợi tại mỗi nút giao.

## 1. Các thành phần Phần thưởng
MGMQ sử dụng ba chỉ số chính để đánh giá hiệu quả của một hành động (pha đèn):

- **$r_{pn}$ (Pass Number)**: Tổng số phương tiện đã rời khỏi nút giao thành công kể từ khi thực hiện hành động cuối cùng. Đây là thành phần mang giá trị dương (phần thưởng).
- **$r_{wn}$ (Waiting Number)**: Tổng số phương tiện đang đứng yên (tốc độ bằng 0) trên tất cả các làn đường đi vào ngõ giao. Đây là thành phần mang giá trị âm (phạt).
- **$r_{wt}$ (Waiting Time)**: Tổng thời gian chờ tích lũy của toàn bộ xe đang dừng trên các làn ngõ vào. Đây là thành phần phạt bổ sung để tránh việc xe phải chờ quá lâu.

## 2. Công thức Hàm thưởng Tổng quát
Phần thưởng $r$ cho một tác nhân tại thời điểm $t$ được tính bằng tổng trọng số của các thành phần trên:

$$r = r_{pn} - \left( r_{wn} + \frac{r_{wt}}{10} \right)$$

Trong đó:
- Hệ số **10** được sử dụng để điều chỉnh độ lớn (rescale) của thời gian chờ, đảm bảo sự cân bằng về trọng số giữa số lượng xe và thời gian.
- **Mục tiêu**: Tối đa hóa số xe đi qua ($r_{pn}$) và tối thiểu hóa hàng đợi ($r_{wn}$) cũng như thời gian trễ ($r_{wt}$).

## 3. Hình phạt Chuyển pha ($r_p$)
Ngoài công thức chính, MGMQ còn cân nhắc đến yếu tố ổn định của dòng giao thông. Một hình phạt bổ sung $r_p$ có thể được áp dụng khi tác nhân thay đổi pha đèn quá thường xuyên:
- $r_p = +1$: Nếu pha đèn tiếp theo giữ nguyên như pha hiện tại.
- $r_p = -1$: Nếu pha đèn thay đổi.
- **Ý nghĩa**: Ngăn chặn tình trạng chuyển pha liên tục (frequent switching), giúp xe tránh phải khởi động và dừng lại quá nhiều lần, từ đó vừa tăng hiệu quả lưu thông vừa giảm nguy cơ tai nạn.

## 4. Đặc điểm nổi bật
Khác với các phương pháp chỉ tập trung vào độ dài hàng đợi, hàm thưởng của MGMQ khuyến khích tác nhân học được chiến lược "xả xe" chủ động thông qua thành phần $r_{pn}$. Điều này giúp hệ thống đạt được trạng thái cân bằng động (dynamic equilibrium) trên toàn mạng lưới đô thị.
