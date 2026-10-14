# Bài 04: K-Láng giềng gần nhất (K-Nearest Neighbors - KNN)

## 1. Tóm tắt lý thuyết

* **Nguyên lý thuật toán (Học lười - Lazy Learning):**
  KNN không có pha tối ưu hàm mất mát hay tìm trọng số trong giai đoạn huấn luyện (`.fit()` chỉ lưu trữ dữ liệu vào bộ nhớ). Quá trình tính toán diễn ra tại thời điểm dự đoán (`.predict()`): với mỗi điểm mới, thuật toán tính khoảng cách tới tất cả các điểm trong tập huấn luyện để chọn ra $k$ điểm lân cận gần nhất.
* **Độ đo khoảng cách:**
  * **Khoảng cách Euclid ($L_2$):** $d(u, v) = \sqrt{\sum_{i=1}^p (u_i - v_i)^2}$, là đường thẳng chim bay trong không gian liên tục.
  * **Khoảng cách Manhattan ($L_1$):** $d(u, v) = \sum_{i=1}^p |u_i - v_i|$, đo theo dạng lưới ô bàn cờ.
* **Tham số $k$ và sự đánh đổi Bias - Variance:**
  * $k$ nhỏ (ví dụ $k=1$): Mô hình rất nhạy cảm với nhiễu, biên phân chia phức tạp (Variance cao, dễ Overfitting).
  * $k$ lớn: Biên phân chia quá mượt, phụ thuộc vào tỷ lệ nhãn lớp đa số (Bias cao, dễ Underfitting).
* **Tầm quan trọng của chuẩn hóa dữ liệu:**
  KNN tính toán dựa trực tiếp trên khoảng cách hình học, do đó các đặc trưng có thang đo lớn sẽ lấn át hoàn toàn các đặc trưng có thang đo nhỏ nếu không qua bước chuẩn hóa (`StandardScaler`).

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Tự cài đặt thuật toán KNN từ scratch**
  * Tự viết hàm tính khoảng cách Euclid, sắp xếp lấy $k$ điểm gần nhất và bỏ phiếu theo đa số bằng NumPy.
  * Đánh giá trên tập dữ liệu kiểm tra: Độ chính xác đạt **$0.9200$**, trùng khớp hoàn toàn với `KNeighborsClassifier` của Scikit-learn.

* **Bài 2: So sánh khoảng cách Euclid và Manhattan**
  * Thử nghiệm với khoảng cách Manhattan ($p=1$): Độ chính xác đạt $0.9067$ (tại $k=27$), chênh lệch không đáng kể so với khoảng cách Euclid đạt $0.9200$ (tại $k=13$).
  * Trong không gian đặc trưng 2 chiều liên tục, thứ tự các điểm láng giềng gần nhau ít bị xáo trộn khi thay đổi giữa 2 thước đo.

* **Bài 3: Khảo sát ảnh hưởng của chuẩn hóa đặc trưng**
  * Khi chưa chuẩn hóa: Đặc trưng chi tiêu có thang đo hàng nghìn lấn át thang điểm 10 của điểm số, độ chính xác chỉ đạt **$53.33\%$** (tương đương đoán ngẫu nhiên).
  * Khi chuyển đổi đơn vị về triệu đồng hoặc chuẩn hóa bằng `StandardScaler`: Độ chính xác tăng vọt lên **$86.67\%$**. Thí nghiệm khẳng định việc chuẩn hóa thang đo là bước bắt buộc khi sử dụng KNN.

* **Bài 4: Trọng số khoảng cách `weights='distance'`**
  * Khi chọn $k$ lớn ($k = 101$), chế độ mặc định `weights='uniform'` làm giảm độ chính xác xuống $85.33\%$ do các điểm ở xa cũng có trọng số biểu quyết ngang với điểm ở gần.
  * Bật chế độ `weights='distance'` (gán trọng số tỉ lệ nghịch với khoảng cách $1/d$): Độ chính xác được phục hồi lên **$90.67\%$**, giúp hạn chế hiện tượng Underfitting khi $k$ lớn.

* **Bài 5: Hạn chế của KNN trong bài toán hồi quy**
  * Áp dụng `KNeighborsRegressor` dự đoán giá nhà từ diện tích (`gia_nha.csv`): Giá trị $k$ tốt nhất chọn qua kiểm định chéo là $k=2$.
  * Sai số MAE của KNN đạt $0.4867$ tỷ đồng, kém hơn so với Hồi quy tuyến tính ($0.2578$ tỷ đồng). Nguyên nhân là dữ liệu có xu hướng tuyến tính rõ ràng, KNN chỉ ước lượng dạng bậc thang cục bộ và không có khả năng ngoại suy ngoài miền dữ liệu huấn luyện.

* **Bài 6: Đánh giá thời gian dự đoán theo kích thước dữ liệu**
  * Khảo sát thời gian thực thi: Giai đoạn `.fit()` chỉ mất vài phần nghìn giây, nhưng thời gian `.predict()` tăng tỉ lệ thuận với kích thước tập dữ liệu huấn luyện $O(n \cdot p)$, thể hiện rõ đặc tính "học lười" của KNN.

---

## 3. Nhận xét tổng kết

* KNN là mô hình phi tham số trực quan, không giả định phân phối dữ liệu, hoạt động tốt trên các bài toán có ranh giới phi tuyến trong không gian số chiều thấp.
* Luôn phải chuẩn hóa dữ liệu trước khi đưa vào mô hình.
* Hạn chế: Thời gian dự đoán và bộ nhớ tăng tuyến tính theo số mẫu; bị ảnh hưởng nặng bởi "lời nguyền số chiều" (Curse of Dimensionality) khi số lượng đặc trưng lớn.
