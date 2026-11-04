# Bài 07: Đánh giá mô hình (Model Evaluation)

## 1. Tóm tắt lý thuyết

* **Vấn đề mất cân bằng dữ liệu (Class Imbalance):**
  Khi một lớp chiếm đa số (ví dụ $90\%$ sinh viên không bỏ học), mô hình chỉ cần dự đoán toàn bộ là lớp đa số vẫn đạt Accuracy $90\%$. Do đó, Accuracy không phải là thước đo đáng tin cậy trên dữ liệu lệch lớp.
* **Ma trận nhầm lẫn (Confusion Matrix):**
  * **TP (True Positive):** Mẫu dương tính thực tế được dự đoán đúng.
  * **TN (True Negative):** Mẫu âm tính thực tế được dự đoán đúng.
  * **FP (False Positive):** Mẫu âm tính nhưng bị báo nhầm thành dương tính (báo động giả).
  * **FN (False Negative):** Mẫu dương tính nhưng bị bỏ sót thành âm tính.
* **Bộ chỉ số đánh giá phân loại:**
  * $\text{Precision} = \frac{TP}{TP + FP}$: Đo lường độ tin cậy khi mô hình dự đoán lớp dương.
  * $\text{Recall} = \frac{TP}{TP + FN}$: Đo lường khả năng thu thập toàn bộ các mẫu dương tính thực tế.
  * $\text{F1-score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$: Trung bình điều hòa giữa Precision và Recall.
* **Đường cong ROC và AUC:**
  * Đường cong ROC biểu diễn mối quan hệ giữa True Positive Rate (Recall) và False Positive Rate qua tất cả các ngưỡng xác suất từ $0$ đến $1$.
  * Chỉ số AUC (Area Under Curve) đo năng lực phân biệt tổng thể của mô hình (AUC = $1.0$ là hoàn hảo, AUC = $0.5$ tương đương đoán ngẫu nhiên).
* **Đánh giá mô hình hồi quy:**
  * **MAE (Mean Absolute Error):** Trung bình sai số tuyệt đối, ít bị ảnh hưởng bởi ngoại lai.
  * **RMSE (Root Mean Squared Error):** Căn bậc hai trung bình bình phương sai số, phạt rất nặng các sai số lớn do có phép bình phương.
  * **$R^2$ Score:** Hệ số xác định, đo tỷ lệ phương sai của biến mục tiêu được giải thích bởi mô hình.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Tự viết hàm đánh giá chỉ dùng NumPy**
  * Tự cài đặt các hàm tính ma trận nhầm lẫn và các thước đo chỉ dùng mảng NumPy trên kết quả dự đoán của dữ liệu `sinh_vien_bo_hoc.csv`.
  * Kết quả thu được: $TP=17, FP=3, FN=13, TN=267$; Accuracy = $0.9467$, Precision = $0.8500$, Recall = $0.5667$, F1 = $0.6800$, hoàn toàn trùng khớp với hàm của thư viện `sklearn.metrics`.

* **Bài 2: Báo cáo phân loại trên bộ chữ số viết tay**
  * Đánh giá chi tiết mô hình trên 10 lớp chữ số của tập Digits.
  * Hầu hết các chữ số đạt Precision và Recall trên $95\%$. Các lớp chữ số có hình dạng tương đồng (như chữ số 1 và 8, hoặc 3 và 9) xuất hiện một vài trường hợp nhầm lẫn nhỏ.

* **Bài 3: Chọn ngưỡng theo yêu cầu Recall $\ge 0.8$**
  * Áp dụng kiểm định chéo (`cross_val_predict`) trên tập huấn luyện để dò ngưỡng xác suất phù hợp với chỉ tiêu bài toán đặt ra (bắt trúng ít nhất $80\%$ sinh viên có nguy cơ bỏ học).
  * Tìm được ngưỡng tối ưu là **$0.15$**. Khi kiểm tra trên tập test độc lập, mô hình đạt Recall = **$0.8333$** (bắt đúng 25/30 sinh viên bỏ học), Precision = $0.5102$ và tổng số trường hợp cảnh báo là 49 sinh viên.

* **Bài 4: Tự vẽ và so sánh đường cong ROC**
  * Huấn luyện và vẽ đường cong ROC cho hai mô hình Logistic Regression và Random Forest trên cùng một đồ thị, lưu vào tệp `roc.png`.
  * Mô hình Hồi quy Logistic đạt chỉ số $\text{AUC} = 0.9449$, cao hơn Random Forest ($\text{AUC} = 0.9249$) do dữ liệu bài toán có ranh giới mang tính chất tuyến tính rõ ràng.

* **Bài 5: Cân bằng lớp cho Random Forest và SVM**
  * Bật tham số `class_weight='balanced'` để tự động tăng mức phạt sai số cho lớp thiểu số (sinh viên bỏ học).
  * Mô hình **SVM RBF có sự cải thiện lớn nhất**: Recall tăng từ $0.5000$ lên **$0.8000$** (+30%), giúp phát hiện được thêm 9 trường hợp bỏ học thực tế so với khi không sử dụng cân bằng lớp.

* **Bài 6: Khảo sát độ nhạy của MAE và RMSE trước ngoại lai**
  * Huấn luyện mô hình Hồi quy tuyến tính trên dữ liệu giá nhà (`gia_nha.csv`), sau đó cố tình tạo một ngoại lai dự đoán lệch 5 tỷ đồng.
  * Khi xuất hiện ngoại lai: Sai số MAE tăng từ $0.2578$ lên $0.6745$ tỷ (tăng $2.6$ lần), trong khi **RMSE tăng từ $0.3730$ lên $1.5505$ tỷ (tăng $4.15$ lần)** và hệ số $R^2$ tụt dốc từ $0.9622$ xuống $0.3475$. Thí nghiệm chứng minh RMSE cực kỳ nhạy cảm với các điểm ngoại lai lớn.

---

## 3. Nhận xét tổng kết

* Khi đánh giá bài toán phân loại trên dữ liệu mất cân bằng, luôn ưu tiên sử dụng Precision, Recall, F1-score hoặc PR-AUC thay vì chỉ nhìn vào Accuracy.
* Ngưỡng phân loại là một công cụ ra quyết định quan trọng, cho phép chủ động điều chỉnh mức độ ưu tiên giữa việc hạn chế bỏ sót (Recall) hay hạn chế báo động giả (Precision).
* Trong bài toán hồi quy, nếu tập dữ liệu có nhiều nhiễu hoặc ngoại lai không thể loại bỏ, MAE là thước đo khách quan và ổn định hơn RMSE.
