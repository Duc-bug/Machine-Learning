# Bài 09: Mạng Nơ-ron Nhân tạo (Multi-Layer Perceptron)

## 1. Tóm tắt lý thuyết

* **Kiến trúc mạng nơ-ron nhiều tầng (MLP):**
  Bao gồm tầng đầu vào (Input layer), một hoặc nhiều tầng ẩn (Hidden layers) và một tầng đầu ra (Output layer). Mỗi nơ-ron thực hiện phép tính tổng có trọng số kèm độ lệch: $z = w^T x + b$, sau đó đưa qua hàm kích hoạt phi tuyến $a = f(z)$.
* **Vai trò của hàm kích hoạt phi tuyến (Activation Function):**
  Nếu không có hàm phi tuyến, dù xếp bao nhiêu tầng nơ-ron thì toàn bộ mạng vẫn chỉ là một tổ hợp tuyến tính đơn thuần.
  * **Sigmoid (Logistic):** Nén giá trị về $(0, 1)$, thường dùng ở tầng ra của bài toán phân loại nhị phân. Nhược điểm: Dễ bị bão hòa gradient (triệt tiêu gradient) ở hai đầu vô cực.
  * **Tanh:** Nén giá trị về $(-1, 1)$, đối xứng qua 0, cải thiện tốc độ hội tụ so với Sigmoid.
  * **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$, tính toán rất nhanh, đạo hàm bằng 1 ở miền dương giúp giảm thiểu triệt để hiện tượng triệt tiêu gradient.
* **Cơ chế huấn luyện:**
  * **Lan truyền thuận (Forward pass):** Tính toán giá trị dự đoán từ tầng vào đến tầng ra.
  * **Lan truyền ngược (Backpropagation):** Sử dụng quy tắc đạo hàm chuỗi (Chain rule) để tính gradient của hàm mất mát theo từng trọng số và cập nhật trọng số bằng thuật toán tối ưu (Gradient Descent hoặc Adam).
* **Kiểm soát Overfitting:**
  * **Hệ số phạt $L_2$ (`alpha`):** Co nhỏ độ lớn của các trọng số để làm mượt ranh giới quyết định.
  * **Early Stopping:** Tự động dừng quá trình huấn luyện khi độ mất mát trên tập xác thực không còn giảm sau một số epoch liên tiếp.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Tính toán thuận thủ công (Forward Pass)**
  * Huấn luyện mô hình `MLPClassifier` với 2 nơ-ron ẩn trên bộ dữ liệu `xor.csv`. Trích xuất ma trận trọng số $W_1, W_2$ và vector bias $b_1, b_2$.
  * Tự cài đặt thuật toán Forward pass bằng NumPy: tính $z_1 = X W_1 + b_1$, kích hoạt ẩn $h = \text{ReLU}(z_1)$, tính $z_2 = h W_2 + b_2$ và xác suất $\hat{p} = \sigma(z_2)$.
  * So sánh với hàm `predict_proba` của Scikit-learn: Sai số lớn nhất giữa hai phương pháp bằng $0.00000000$, chứng minh tính chính xác tuyệt đối của công thức toán học.

* **Bài 2: Thiết kế mạng nơ-ron thủ công cho hàm XOR bằng ReLU**
  * Thiết kế kiến trúc mạng giải quyết bài toán XOR kinh điển:
    * Nơ-ron ẩn 1: $h_1 = \max(0, x_1 - x_2)$
    * Nơ-ron ẩn 2: $h_2 = \max(0, x_2 - x_1)$
    * Nơ-ron ra: $y = 1 \cdot h_1 + 1 \cdot h_2 + 0$
  * Kiểm thử trên 4 mẫu đầu vào của hàm XOR:
    * $[0, 0] \implies h_1=0, h_2=0 \implies y=0$
    * $[0, 1] \implies h_1=0, h_2=1 \implies y=1$
    * $[1, 0] \implies h_1=1, h_2=0 \implies y=1$
    * $[1, 1] \implies h_1=0, h_2=0 \implies y=0$
    * Kết quả khớp $100\%$ với bảng chân trị hàm XOR.

* **Bài 3: So sánh các hàm kích hoạt trên dữ liệu phi tuyến**
  * Đánh giá 3 hàm kích hoạt `logistic`, `tanh` và `relu` với cùng kiến trúc `hidden_layer_sizes=(20, 20)` trên tập dữ liệu hai vầng trăng.
  * Hàm `relu` hội tụ nhanh nhất và cho đường ranh giới phân tách mượt mà với độ chính xác kiểm tra đạt $92.22\%$. Các hàm `logistic` và `tanh` tốn nhiều epoch hơn để đạt độ chính xác tương đương.

* **Bài 4: Thử nghiệm tốc độ học (Learning Rate Init)**
  * Khảo sát các giá trị `learning_rate_init` $\in [0.0001, 0.001, 0.01, 0.1]$ với thuật toán tối ưu Adam.
  * Tốc độ học $\eta = 0.0001$ hội tụ rất chậm, cần nhiều vòng lặp; tốc độ học lớn $\eta = 0.1$ làm hàm mất mát dao động mạnh ở những epoch đầu. Khoảng giá trị $\eta = 0.001 \to 0.01$ cho tốc độ giảm loss mượt mà và ổn định nhất. Đã lưu biểu đồ vào tệp `toc_do_hoc.png`.

* **Bài 5: Tinh chỉnh siêu tham số trên tập Digits bằng GridSearchCV**
  * Dùng `GridSearchCV` kết hợp pipeline chuẩn hóa để tìm kiếm lưới giữa `hidden_layer_sizes` và `alpha`.
  * Cấu hình tối ưu tìm được là: `hidden_layer_sizes=(100,)` và `alpha=1.0`, đạt độ chính xác kiểm định chéo $97.77\%$ và độ chính xác trên tập test độc lập đạt **$98.44\%$**.

* **Bài 6: Mạng nơ-ron hồi quy và khả năng ngoại suy**
  * Huấn luyện `MLPRegressor` trên bài toán dự đoán giá nhà từ diện tích (`gia_nha.csv`).
  * Mô hình với hàm kích hoạt ReLU xấp xỉ hàm mục tiêu bằng các đoạn thẳng tuyến tính từng khúc (piecewise linear). Khi dự đoán cho căn nhà $250\text{ m}^2$ (vượt ngoài dữ liệu huấn luyện $\le 150\text{ m}^2$), MLP tiếp tục ngoại suy tuyến tính theo độ dốc của đoạn cuối cùng (ước tính $\approx 28.5$ tỷ đồng).
  * Điều này khác biệt hoàn toàn với Cây quyết định / Random Forest (chỉ dự đoán giá trị trung bình nằm ngang của nút lá ngoài cùng, hoàn toàn không có khả năng ngoại suy tăng tiếp).

---

## 3. Nhận xét tổng kết

* Mạng nơ-ron nhân tạo (MLP) là mô hình xấp xỉ hàm vạn năng (Universal Approximation Theorem), có khả năng học các mối quan hệ phi tuyến rất phức tạp khi có đủ dữ liệu và tầng ẩn.
* Bắt buộc phải chuẩn hóa đặc trưng (bằng `StandardScaler`) trước khi huấn luyện MLP để tránh hiện tượng mất cân bằng thang đo gây nghẽn quá trình cập nhật gradient.
* Hạn chế: MLP có số lượng tham số lớn, chi phí tính toán cao và là mô hình dạng "hộp đen" (black-box), khó diễn giải ý nghĩa của từng trọng số so với các mô hình tuyến tính hoặc cây quyết định.
