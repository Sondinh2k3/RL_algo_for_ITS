# Khung Huấn luyện và Hàm Thưởng của GESA

Dựa trên tài liệu: *2024 - A general scenario-agnostic reinforcement learning for traffic signal control.pdf*

GESA sử dụng khung học tăng cường Actor-Critic cải tiến, được thiết kế để huấn luyện đồng thời trên quy mô lớn.

## 1. Khung học tập (Learning Framework)
GESA tích hợp thiết kế thống nhất vào các mô hình RL hiện đại để xử lý các nút giao tùy ý.

- **Thuật toán chính**: Sử dụng Proximal Policy Optimization (PPO) kết hợp Asynchronous Advantage Actor-Critic (A3C).
- **Cơ chế Co-training**: Mỗi tiến trình (process) nhận một kịch bản để tương tác với mô hình và trả về gradient cập nhật. Điều này giúp mô hình tiếp xúc với sự đa dạng của các cấu trúc nút giao trong quá trình học.

## 2. Kiến trúc Mạng (Network Design)
Kiến trúc GESA cải tiến mô hình FRAP nguyên bản (vốn sử dụng DQN) sang dạng Actor-Critic để xử lý các kịch bản hành động liên tục và thân thiện hơn với huấn luyện song song.

- **FRAP Module cải tiến**:
    - Nhúng pha và cặp pha để nắm bắt quan hệ cạnh tranh.
    - Sử dụng lớp Convolution 1x1 và tích Kronecker với mặt nạ cạnh tranh ($\Omega$) để tạo ra nhúng cặp pha có mặt nạ (masked phase-pair embedding).
- **Actor Network**: Xuất ra phân phối hành động $\pi(a_i|f_i)$ thông qua lớp Softmax để chọn pha.
- **Critic Network**: Dự báo giá trị trạng thái $V(f_i)$.

## 3. Hệ thống Hàm thưởng (Reward System)
Phần thưởng $R_t$ tại mỗi bước $t$ là tổng trọng số của 4 thành phần chính nhằm tối thiểu hóa ùn tắc:
$$R_t = \sum_{c \in C} w_c c_i$$

### Các thành phần và Trọng số:
Để tối đa hóa phần thưởng tổng, các trọng số được đặt giá trị âm (trừng phạt):

| Thành phần | Ký hiệu | Trọng số ($w_c$) | Mục tiêu |
| :--- | :--- | :--- | :--- |
| **Delay time** | $T^d_i$ | $-1\times 10^{-5}$ | Giảm độ trễ thời gian |
| **Wait time** | $T^w_i$ | $-1\times 10^{-3}$ | Giảm thời gian chờ trung bình |
| **Queue length**| $L_i$ | $-1\times 10^{-3}$ | Giảm độ dài hàng đợi |
| **Pressure** | $P_i$ | $-5\times 10^{-3}$ | Cân bằng áp lực luồng vào/ra |

## 4. Kết quả Thực nghiệm
- **Hiệu quả Co-training**: Việc huấn luyện đa kịch bản giúp mô hình hội tụ nhanh hơn và ổn định hơn so với huấn luyện trên một kịch bản đơn lẻ (tốc độ hội tụ nhanh hơn rõ rệt trong các kịch bản phức tạp như Nanshan hay Fenglin).
- **Tính tổng quát**: GESA thể hiện sức mạnh vượt trội khi đối đầu với các phương pháp truyền thống như MaxPressure hay các phương pháp RL như MetaLight trong cả kịch bản đã biết và kịch bản mới hoàn toàn.
