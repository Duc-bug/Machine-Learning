# Bài 03: Phân loại Naive Bayes (Naive Bayes Classifier)

## 1. Tóm tắt lý thuyết

* **Định lý Bayes:**
  $$P(y|x) = \frac{P(x|y) P(y)}{P(x)}$$
  Trong đó: $P(y)$ là xác suất tiên nghiệm (prior), $P(x|y)$ là hợp lý (likelihood), $P(y|x)$ là xác suất hậu nghiệm (posterior).
* **Giả định độc lập có điều kiện ("Naive"):**
  Giả sử các đặc trưng $x_1, x_2, \dots, x_n$ độc lập với nhau khi biết trước nhãn lớp $y$:
  $$P(x_1, x_2, \dots, x_n | y) = \prod_{i=1}^n P(x_i | y)$$
* **Biến đổi hàm Log tránh tràn dưới (Underflow):**
  Nhân nhiều xác suất nhỏ liên tiếp dễ dẫn tới lỗi tràn dưới số thực trong máy tính. Chuyển sang không gian Log giúp biến tích thành tổng:
  $$\log P(y|x) \propto \log P(y) + \sum_{i=1}^n \log P(x_i | y)$$
* **Laplace Smoothing (Làm trơn Laplace):**
  $$\hat{P}(x_i | y) = \frac{N_{yi} + \alpha}{N_y + \alpha d}$$
  Thêm hệ số $\alpha > 0$ vào tử và mẫu giúp khắc phục lỗi xác suất bằng 0 khi gặp từ vựng chưa từng xuất hiện trong tập huấn luyện.
* **Các biến thể chính:**
  * **MultinomialNB:** Phù hợp dữ liệu đếm từ / tần suất (bài toán phân loại văn bản, lọc thư rác).
  * **GaussianNB:** Phù hợp dữ liệu dạng số thực liên tục có phân phối xấp xỉ chuẩn.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Bayes với ba nguyên nhân**
  * Lớp sinh viên gồm 3 nhóm: học đều ($50\%$, tỷ lệ qua $0.95$), học dồn ($30\%$, tỷ lệ qua $0.70$), ít học ($20\%$, tỷ lệ qua $0.30$).
  * Xác suất tổng quát qua môn: $P(\text{Qua}) = 0.5 \times 0.95 + 0.3 \times 0.70 + 0.2 \times 0.30 = 0.745$.
  * Áp dụng công thức Bayes khi biết một sinh viên đã qua môn:
    * Xác suất thuộc nhóm học đều: $P(\text{Học đều}|\text{Qua}) = \frac{0.5 \times 0.95}{0.745} \approx 63.76\%$.
    * Xác suất thuộc nhóm học dồn: $P(\text{Học dồn}|\text{Qua}) = \frac{0.3 \times 0.70}{0.745} \approx 28.19\%$.
    * Xác suất thuộc nhóm ít học: $P(\text{Ít học}|\text{Qua}) = \frac{0.2 \times 0.30}{0.745} \approx 8.05\%$.
    * Tổng xác suất ba nhóm: $63.76\% + 28.19\% + 8.05\% = 100\%$.

* **Bài 2: Xác định từ tố cáo mạnh nhất**
  * Huấn luyện `MultinomialNB` trên 300 tin nhắn (`tin_nhan.csv`). Dùng thuộc tính `feature_log_prob_` tính hiệu log-prob giữa lớp rác và lớp thường: $\Delta = \log P(w|\text{rác}) - \log P(w|\text{thường})$.
  * 5 từ tố cáo tin rác mạnh nhất: `khuyen_mai`, `mien_phi`, `trung_thuong`, `voucher`, `qua_tang`.
  * 5 từ đặc trưng cho tin thường: `hoc`, `diem`, `lop`, `thi`, `nhiem_vu`.

* **Bài 3: Khảo sát tham số làm trơn $\alpha$**
  * Chạy thử nghiệm với $\alpha \in [0.001, 0.1, 1.0, 10.0, 100.0]$.
  * Với $\alpha$ từ $0.001$ đến $1.0$, các chỉ số trên tập kiểm tra ổn định ở mức cao (Accuracy = $0.9467$, Precision = $1.0000$, Recall = $0.8667$).
  * Khi $\alpha$ quá lớn ($100.0$), xác suất của các từ bị kéo đều về nhau, làm giảm độ phân biệt đặc trưng giữa hai lớp.

* **Bài 4: Tự viết bộ lọc không dùng thư viện**
  * Tự cài đặt class Naive Bayes bằng NumPy: đếm từ vựng, tính tiên nghiệm $P(y)$, tính likelihood có Laplace smoothing và cộng tổng log-likelihood khi dự đoán.
  * Đánh giá trên tập kiểm tra (chia bằng `random_state=17`): Thuật toán tự code cho kết quả Accuracy = $0.9467$, khớp hoàn toàn với `MultinomialNB` của Scikit-learn.

* **Bài 5: Chọn sai biến thể Naive Bayes**
  * Chạy `MultinomialNB` trên dữ liệu điểm sinh viên liên tục (`gio_on` và `diem_giua_ky`).
  * Chương trình không báo lỗi vì Scikit-learn chấp nhận số thực dương. Tuy nhiên, `MultinomialNB` giả định dữ liệu đếm rời rạc nên kết quả không phù hợp về mặt lý thuyết so với `GaussianNB`.

* **Bài 6: Lớp hiếm và xác suất tiên nghiệm**
  * Tạo tập dữ liệu mất cân bằng: giữ nguyên 180 tin thường và chỉ lấy 20 tin rác.
  * Mô hình đạt Accuracy rất cao ($98.00\%$), nhưng Recall của lớp tin rác bị kéo giảm đáng kể do xác suất tiên nghiệm $P(\text{rác})$ quá nhỏ ($10\%$). Điều này minh họa cho nghịch lý Accuracy khi dữ liệu bị lệch lớp.

---

## 3. Nhận xét tổng kết

* Naive Bayes có cấu trúc tính toán đơn giản, tốc độ huấn luyện và suy luận nhanh, bộ nhớ thấp, rất phù hợp với bài toán phân loại văn bản có nhiều chiều từ vựng.
* Kỹ thuật biến đổi sang miền Log và làm trơn Laplace là hai yếu tố quan trọng giúp mô hình hoạt động ổn định trên dữ liệu thực tế.
* Hạn chế: Giả định độc lập có điều kiện giữa các biến thường không thỏa mãn trọn vẹn trong ngôn ngữ tự nhiên; mô hình dễ bị ảnh hưởng bởi tỷ lệ mất cân bằng lớp nếu không hiệu chỉnh ngưỡng.
