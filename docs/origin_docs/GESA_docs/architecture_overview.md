# Tổng quan Kiến trúc GESA (General Scenario-Agnostic)

Dựa trên tài liệu: *2024 - A general scenario-agnostic reinforcement learning for traffic signal control.pdf*

## 1. Mục tiêu (Objective)
GESA được thiết kế để giải quyết thách thức **sim2real** (từ mô phỏng ra thực tế) và tính **tổng quát** (generalization) trong điều khiển tín hiệu giao thông (TSC). Các phương pháp hiện tại thường bị ràng buộc với các kịch bản cụ thể, đòi hỏi nhiều công sức gán nhãn thủ công cho cấu trúc nút giao. GESA loại bỏ việc gán nhãn thủ công và cho phép huấn luyện đồng thời trên nhiều kịch bản khác nhau.

## 2. Các thành phần lõi (Core Components)

### A. Mô-đun Plug-in Tổng quát (General Plug-in Module - GPI)
- Chức năng: Ánh xạ tất cả các nút giao khác nhau vào một cấu trúc thống nhất (unified structure).
- Giúp giải phóng khâu gán nhãn thủ công cấu trúc nút giao.
- Sử dụng các vector vị trí của hướng tiếp cận để chuẩn hóa các nút giao phức tạp về nút giao chuẩn.

### B. Không gian Trạng thái và Hành động Thống nhất (Unified State and Action Space)
- Giữ cho đầu vào (input) và đầu ra (output) của mô hình có cấu trúc nhất quán.
- Cho phép mô hình xử lý nhiều nút giao có cấu hình khác nhau mà không cần thay đổi kiến trúc mạng thần kinh.

### C. Đồng huấn luyện quy mô lớn (Large-scale Co-training)
- Huấn luyện đồng thời trên nhiều kịch bản (ví dụ: Grid 4x4, Arterial, Ingolstadt, Fenglin, Nanshan).
- Tạo ra một thuật toán điều khiển tín hiệu giao thông có tính chất tổng quát cao.

## 3. Quy trình xử lý
1. **Unify Approaches**: Sử dụng Algorithm 1 để căn chỉnh hướng của các tiếp cận (North, East, South, West) dựa trên góc tương đối.
2. **Unify Movements**: Ánh xạ các làn đường vào 8 chuyển động giao thông chuẩn (Thông qua và Rẽ trái cho 4 hướng).
3. **Feature Fusion**: Kết hợp các đặc trưng chuyển động để tạo ra nhúng pha (phase embedding).
4. **Decision**: Sử dụng mạng Actor-Critic (được cải tiến từ FRAP) để đưa ra quyết định chọn pha.

## 4. Hiệu quả
- Đạt phần thưởng cao hơn 13.27% so với các phương pháp benchmark khi đồng huấn luyện.
- Đạt hiệu quả vượt trội (9.39% cao hơn) khi áp dụng vào kịch bản chưa từng gặp (zero-shot transfer).
