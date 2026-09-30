# Bài 02: Hồi quy Logistic (Logistic Regression)

## 1. Tóm tắt lý thuyết

* **Bản chất bài toán:** Hồi quy Logistic dùng cho bài toán phân loại nhị phân ($y \in \{0, 1\}$), ước lượng xác suất mẫu thuộc về lớp dương: $P(y=1|x) = \hat{p}$.
* **Hàm kích hoạt Sigmoid:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{với } z = w^T x + b$$
  Hàm Sigmoid ánh xạ toàn bộ miền giá trị thực vào khoảng $(0, 1)$. Tại $z = 0$, $\sigma(0) = 0.5$ (ranh giới phân vân 50/50).
* **Ranh giới quyết định (Decision Boundary):** Siêu phẳng thỏa mãn $w^T x + b = 0$, phân tách không gian thành hai miền dự đoán nhãn 0 và nhãn 1.
* **Hàm mất mát Binary Cross-Entropy (Log Loss):**
  $$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \Big[ y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \Big]$$
  Hàm mất mát là hàm lồi, đảm bảo thuật toán Gradient Descent hội tụ về cực tiểu toàn cục.
* **Các chỉ số đánh giá phân loại:**
  * **Accuracy:** Tỷ lệ dự đoán đúng tổng thể, không phù hợp khi dữ liệu mất cân bằng.
  * **Precision:** Tỷ lệ dự đoán dương tính thực sự đúng ($TP / (TP + FP)$), chú trọng giảm báo động giả.
  * **Recall:** Tỷ lệ bắt trúng các mẫu dương tính thực tế ($TP / (TP + FN)$), chú trọng tránh bỏ sót.
  * **F1-score:** Trung bình điều hòa giữa Precision và Recall.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Thống kê theo nhóm điểm giữa kỳ**
  * Phân tích bộ dữ liệu sinh viên (`sinh_vien.csv`, 120 mẫu): Có 53 sinh viên đạt điểm giữa kỳ $\ge 7$.
  * Tỷ lệ qua môn của nhóm này đạt $94.34\%$ (50/53 bạn qua môn), cho thấy điểm giữa kỳ là đặc trưng phân hóa rất mạnh.

* **Bài 2: Vẽ hàm Sigmoid**
  * Vẽ đồ thị hàm Sigmoid với $z \in [-8, 8]$ và tiếp tuyến tại điểm đối xứng $z=0$ (độ dốc $0.25$).
  * Đồ thị được lưu tại tệp `sigmoid.png`, minh họa tính chất nén giá trị và tốc độ biến thiên nhanh nhất ở lân cận ranh giới 0.

* **Bài 3: Dự đoán xác suất cho sinh viên cụ thể**
  * Xây dựng hàm `du_doan(gio)` dựa trên mô hình 1 biến: $\hat{p} = \sigma(0.3934 \times \text{gio} - 5.0652)$.
  * Điểm hòa vốn xác suất 50% nằm tại $x = 5.0652 / 0.3934 \approx 12.89$ giờ. Sinh viên ôn tập từ 13 giờ trở lên có xác suất qua môn $> 50\%$.

* **Bài 4: Tự tính bốn thước đo không dùng thư viện**
  * Tự cài đặt các hàm tính ma trận nhầm lẫn ($TP, FP, FN, TN$) và các thước đo Accuracy, Precision, Recall, F1 chỉ dùng mảng NumPy.
  * Kết quả tính tay: Accuracy = $0.9000$, Precision = $0.8889$, Recall = $0.9412$, F1 = $0.9143$, hoàn toàn trùng khớp với hàm có sẵn của Scikit-learn.

* **Bài 5: Dò ngưỡng quyết định tối ưu theo F1**
  * Khảo sát ngưỡng phân loại $t$ từ $0.05$ đến $0.95$ với bước nhảy $0.05$.
  * Ngưỡng mặc định $0.5$ cho $F_1 = 0.8947$. Ngưỡng tối ưu đạt được tại **$t = 0.60$** với **$F_1 = 0.9412$** (Precision đạt $1.0000$, Recall giữ ở $0.8889$, loại bỏ hoàn toàn các ca báo động giả).

* **Bài 6: Đổi lớp dương sang rớt môn**
  * Khi đảo lớp quan tâm sang sinh viên rớt môn (lớp 0), Recall đổi từ $94.12\%$ thành $75.00\%$.
  * Rút ra: Việc chọn nhãn lớp dương quyết định trực tiếp tới ý nghĩa thực tế của thước đo Recall (cần tập trung vào nhóm sinh viên có nguy cơ rớt môn hay nhóm chắc chắn qua môn).

---

## 3. Nhận xét tổng kết

* Hồi quy Logistic là mô hình phân loại nhị phân cơ bản, tính toán nhanh và cung cấp xác suất đầu ra rõ ràng, dễ giải thích trọng số.
* Ngưỡng phân loại mặc định $0.5$ có thể tinh chỉnh linh hoạt tùy theo mục tiêu bài toán (ưu tiên phát hiện tối đa hay ưu tiên độ chính xác dự đoán).
* Hạn chế: Mô hình chỉ tạo ra ranh giới quyết định tuyến tính, không xử lý tốt dữ liệu có quan hệ phi tuyến phức tạp nếu không có bước tạo đặc trưng đa thức.
