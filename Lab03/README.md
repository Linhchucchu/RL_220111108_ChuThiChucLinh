# Lab03 - Monte Carlo Methods

## Thông tin sinh viên
- Họ tên: Chu Thi Chúc Linh
- MSSV: 22011108
- Lớp: K16-AIRB

## Môi trường đã chọn
- **Môi trường chính:** `Blackjack-v1`
  - *Lý do lựa chọn:* Phù hợp tự nhiên với các bài toán đánh giá và điều khiển dựa trên tập episode kết thúc (episodic tasks) mà không cần biết trước mô hình xác suất chuyển trạng thái (model-free)[cite: 2, 3].
- **Môi trường phụ 1:** `FrozenLake-v1`
  - *Lý do lựa chọn:* Kiểm chứng các hàm sinh episode, tính return và First-Visit MC Prediction trên không gian trạng thái lưới (grid world).
- **Môi trường phụ 2:** `Taxi-v3`
  - *Lý do lựa chọn:* Kiểm tra khả năng mở rộng của thuật toán Monte Carlo Prediction trên môi trường có không gian trạng thái lớn hơn (500 states).

## Nội dung thực hiện
- Xây dựng hệ thống sinh episode và trích xuất trajectory.
- Cài đặt tính toán discounted returns $G_t$ với hệ số chiết khấu $\gamma$.
- Cài đặt thuật toán First-Visit và Every-Visit Monte Carlo Prediction ước lượng hàm giá trị trạng thái $V(s)[cite: 2, 3]$.
- Cài đặt ước lượng hàm giá trị hành động $Q(s,a)$ và chính sách khám phá $\epsilon$-greedy[cite: 2, 3].
- Cài đặt giải thuật On-policy First-Visit MC Control để tối ưu hóa chính sách trên `Blackjack-v1`[cite: 2, 3].
- Thử nghiệm các hàm Monte Carlo trên hai môi trường phụ `FrozenLake-v1` và `Taxi-v3`.
- Trực quan hóa đường cong học tập (Learning Curve) và so sánh hiệu năng giữa các policy.
