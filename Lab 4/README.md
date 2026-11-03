## 1. Bản chất của KNN (Học lười - Lazy Learning)
* Lúc gọi `.fit()` mô hình không hề tối ưu hàm mất mát hay tìm trọng số gì cả, nó chỉ đơn thuần cất toàn bộ dữ liệu vào RAM (rất lười).
* Gánh nặng dồn hết vào lúc `.predict()`: với mỗi điểm mới, nó phải đo khoảng cách tới từng điểm trong tập train để tìm hàng xóm. Dữ liệu càng nhiều thì lúc dự đoán càng chậm (thấy rất rõ ở **Bài 6**: fit chỉ tốn vài ms nhưng predict tăng thời gian theo cỡ tập dữ liệu).

## 2. Nhận xét

* **Tự code KNN bằng NumPy :**
  * Tự viết hàm tính khoảng cách Euclid, sort lấy $k$ điểm gần nhất rồi vote nhãn đa số. Kết quả accuracy đạt **0.9200**, giống y hệt thư viện scikit-learn. Bản chất KNN rất trực quan và mang tính hình học thuần túy.

* **Khoảng cách Euclid vs Manhattan (Bài 2):**
  * Thử đổi sang đo kiểu bàn cờ Manhattan ($L_1$), kết quả tập test đạt **0.9067** (với $k=27$), không chênh lệch nhiều so với Euclid **0.9200** (với $k=13$). Trong không gian 2D liên tục, thứ tự các điểm gần nhau gần như không bị đảo lộn nhiều khi đổi thước đo.

* **Bài học về Chuẩn hoá dữ liệu (Bài 3):**
  * *Chưa chuẩn hoá:* Cột chi tiêu dùng đơn vị "nghìn đồng" (giá trị hàng nghìn) làm mô hình chỉ đạt **53.33%** (thua cả đoán bừa). Chỉ cần đổi đơn vị sang "triệu đồng" để thu nhỏ số lại, độ chính xác tăng vọt lên **86.67%**.
  * *Khi có chuẩn hoá:* Cả hai đơn vị đều cho kết quả chuẩn **86.67%**. Rút ra là: **Dùng KNN bắt buộc phải chuẩn hoá z-score**, nếu không đặc trưng nào số to sẽ đè bẹp hết các đặc trưng quan trọng khác.

* **Trọng số khoảng cách `weights='distance'` (Bài 4):**
  * Khi chọn $k$ quá lớn ($k = 101$), việc để `uniform` khiến mô hình bị cẩu thả (underfitting), accuracy rớt xuống **85.33%** vì các điểm ở xa cũng có 1 phiếu ngang điểm ở gần.
  * Khi bật `weights='distance'`, accuracy kéo lại được **90.67%**. Trọng số $1/d$ giúp các điểm nằm thật sự sát có tiếng nói quyết định hơn hẳn.

* **Hạn chế của KNN khi hồi quy (Bài 5):**
  * Trên dữ liệu giá nhà, chọn được $k=2$ tốt nhất qua Cross-Validation nhưng sai số MAE (**0.4867 tỷ**) vẫn thua xa Hồi quy tuyến tính (**0.2578 tỷ**).
  * Lý do: Giá nhà tăng đều theo diện tích, đường thẳng bắt đúng xu hướng toàn cục này. KNN chỉ dự đoán kiểu bậc thang cục bộ và không thể ngoại suy ra ngoài phạm vi dữ liệu đã biết.

---

## 3. Tóm lại: Khi nào nên / không nên dùng KNN?
* **Nên dùng:** Dữ liệu ít chiều, ranh giới phân lớp cong queo phức tạp (như dữ liệu hai vầng trăng), cần mô hình đơn giản không cần huấn luyện cầu kỳ.
* **Không nên dùng:** Dữ liệu có nhiều chiều (bị lời nguyền số chiều), dữ liệu quá lớn cần suy luận nhanh thời gian thực, hoặc bài toán hồi quy có xu hướng tuyến tính rõ ràng.
