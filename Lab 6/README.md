# Bài 06: Máy vectơ hỗ trợ (SVM - Support Vector Machine)

## 1. Tóm tắt lý thuyết

* **Nguyên lý lề cực đại (Maximum Margin):** Khác với Hồi quy Logistic tìm bất kỳ siêu phẳng phân chia nào, SVM tìm siêu phẳng có khoảng cách tới các điểm dữ liệu gần nhất của hai lớp là lớn nhất (độ rộng lề $2 / \|w\|_2$ đạt cực đại).
* **Vectơ hỗ trợ (Support Vectors):** Là tập hợp các điểm dữ liệu nằm ngay trên mép lề ($|w^T x + b| = 1$) hoặc vi phạm lề. Vị trí của siêu phẳng phân chia hoàn toàn do các vectơ hỗ trợ này quyết định; việc thêm bớt các điểm nằm sâu bên trong cụm không làm thay đổi ranh giới.
* **Tham số phạt $C$ (Lề mềm - Soft Margin):**
  * $C$ nhỏ: Phạt nhẹ các điểm vi phạm, ưu tiên lề rộng, chấp nhận một số điểm nằm sai phía (giảm Overfitting).
  * $C$ lớn: Phạt nặng các điểm vi phạm, lề hẹp lại, cố gắng phân chia đúng tối đa các mẫu huấn luyện (dễ dẫn tới Overfitting).
* **Thủ thuật Kernel (Kernel Trick):** Ánh xạ dữ liệu từ không gian ban đầu lên không gian nhiều chiều hơn để phân tách tuyến tính mà không cần tính toán trực tiếp tọa độ trong không gian mới.
  * **Kernel RBF (Radial Basis Function):** Phổ biến nhất cho dữ liệu phi tuyến, được điều khiển bởi tham số $\gamma$ (gamma) xác định bán kính ảnh hưởng của từng mẫu.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Xác định vectơ hỗ trợ và khoảng cách lề**
  * Huấn luyện mô hình lề cứng trên 24 điểm dữ liệu phân tách hoàn toàn.
  * Mô hình tìm ra đúng 3 điểm làm vectơ hỗ trợ. Khoảng cách hình học từ mỗi vectơ hỗ trợ tới siêu phẳng phân chia bằng đúng một nửa độ rộng lề: $d = 1.1063 = 2.2126 / 2$.
  * Khi loại bỏ 21 điểm còn lại và chỉ giữ lại 3 vectơ hỗ trợ để huấn luyện, siêu phẳng phân chia nhận được hoàn toàn trùng khớp với ban đầu ($w_1 = -0.8310, w_2 = -0.3550, b = 4.8200$).

* **Bài 2: Khảo sát ảnh hưởng khi thêm điểm mới**
  * Khi thêm điểm $(-1, -1)$ nằm sâu trong cụm của lớp, siêu phẳng và các vectơ hỗ trợ không thay đổi.
  * Khi thêm điểm $(4.5, 4.0)$ nằm sát vùng tranh chấp, điểm này trở thành vectơ hỗ trợ mới, làm siêu phẳng phân chia xoay lệch để điều chỉnh lại độ rộng lề.

* **Bài 3: Thử nghiệm Kernel đa thức**
  * Đánh giá Kernel đa thức từ bậc 2 đến bậc 5 trên tập dữ liệu hai vầng trăng (`hai_vang_trang.csv`).
  * Bậc 3 cho kết quả tốt nhất trong các đa thức với Accuracy = $82.67\%$, tuy nhiên vẫn kém hơn Kernel RBF (đạt $92.00\%$) do biên đa thức khó uốn cong khép kín theo hình lưỡi liềm. Các bậc chẵn (2, 4) chỉ đạt độ chính xác xấp xỉ $50\%$ do tính đối xứng không phù hợp với phân bố dữ liệu.

* **Bài 4: Tinh chỉnh tham số C và gamma bằng lưới mịn**
  * Sử dụng `GridSearchCV` với 5-fold cross-validation dò chi tiết quanh vùng tối ưu: $C \in [3, 10, 30]$ và $\gamma \in [0.3, 1.0, 3.0]$.
  * Cặp tham số tốt nhất tìm được là $C = 30$ và $\gamma = 1.0$, nâng điểm kiểm định chéo lên **$0.8756$** (so với $0.8711$ của lưới thô $C=10, \gamma=1.0$).

* **Bài 5: Phân tích các trường hợp nhận dạng sai trên tập Digits**
  * Mô hình SVM RBF nhận dạng chữ số viết tay đạt Accuracy $99.11\%$ trên tập kiểm tra (chỉ dự đoán sai 4 trên 450 mẫu).
  * Trực quan hóa 4 mẫu đoán sai cho thấy nét viết thực tế bị dị dạng hoặc đứt nét (số 4 khép kín nhìn giống số 8, số 8 bị ép dẹt giống số 1, số 7 có móc giống số 9, số 9 hở vòng giống số 3).
  * Việc chuẩn hóa đặc trưng bằng `StandardScaler` trên ảnh 8x8 làm giảm nhẹ độ chính xác (xuống $97.78\%$) do các pixel viền ngoài có phương sai xấp xỉ 0 bị phóng đại nhiễu khi chia cho độ lệch chuẩn.

* **Bài 6: Đánh giá độ phức tạp thuật toán khi dữ liệu lớn**
  * So sánh thời gian chạy giữa `LinearSVC` (tuyến tính) và `SVC(kernel='rbf')` trên các tập dữ liệu tổng hợp từ 1,000 đến 10,000 mẫu.
  * `LinearSVC` có độ phức tạp xấp xỉ tuyến tính $O(n)$, thời gian chạy tăng tỉ lệ thuận với số mẫu.
  * `SVC(kernel='rbf')` có độ phức tạp tính toán khoảng $O(n^2)$ đến $O(n^3)$ do phải tính ma trận Gram. Khi số lượng mẫu tăng từ 5,000 lên 10,000, thời gian huấn luyện của RBF tăng hơn 12 lần.

---

## 3. Nhận xét tổng kết

* SVM là mô hình phân loại rất mạnh trên dữ liệu có số chiều cao ($p \ge n$) và các bài toán phân tách phi tuyến phức tạp khi sử dụng Kernel RBF.
* Chuẩn hóa dữ liệu là bước bắt buộc đối với SVM trên các tập dữ liệu dạng bảng có các đặc trưng khác thang đo.
* Đối với tập dữ liệu kích thước lớn (hàng chục nghìn mẫu trở lên), nên ưu tiên dùng `LinearSVC` hoặc các mô hình Gradient Boosting để tránh tắc nghẽn thời gian tính toán của Kernel phi tuyến.
