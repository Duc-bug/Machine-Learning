# Bài 01: Hồi quy Tuyến tính (Linear Regression)

## 1. Tóm tắt kiến thức đã học

* **Bài toán hồi quy:** Dự đoán một giá trị liên tục (giá nhà, nhiệt độ, doanh thu...) thay vì dự đoán nhãn rời rạc như bài toán phân loại.
* **Mô hình tuyến tính & Hàm mất mát MSE:**
  * Giả định quan hệ tuyến tính: $\hat{y} = w^T x + b$.
  * Tối thiểu hóa sai số bình phương trung bình (MSE). MSE có nghiệm đóng và tương đương ước lượng hợp lý cực đại (MLE) khi nhiễu là Gaussian, nhưng rất nhạy cảm với outlier.
* **Hai phương pháp tìm nghiệm:**
  * **Normal Equation (Công thức đóng):** $\hat{\theta} = (X^T X)^{-1} X^T y$, ra kết quả ngay lập tức, phù hợp khi số lượng đặc trưng không quá lớn ($n \lesssim 10.000$).
  * **Gradient Descent (Thuật toán lặp):** Cập nhật trọng số theo hướng ngược gradient, cần chuẩn hóa dữ liệu (`StandardScaler`) và chọn tốc độ học (learning rate) phù hợp để hội tụ nhanh.
* **Đánh giá mô hình:**
  * **RMSE, MAE:** Đo mức độ sai lệch trung bình (cùng đơn vị với $y$).
  * **$R^2$ & Adjusted $R^2$:** Đo tỷ lệ phương sai mô hình giải thích được; Adjusted $R^2$ dùng để phạt bớt các đặc trưng thừa vô nghĩa.
  * **Residual Plot:** Vẽ đồ thị phần dư để kiểm tra các giả định tuyến tính, phương sai không đổi và phân phối chuẩn.
* **Regularization (Ridge & Lasso):**
  * **Ridge ($L_2$):** Phạt bình phương trọng số, giúp co nhỏ hệ số và giải quyết hiện tượng đa cộng tuyến (multicollinearity).
  * **Lasso ($L_1$):** Phạt trị tuyệt đối trọng số, có khả năng kéo hệ số về đúng bằng 0, giúp tự động chọn lọc các đặc trưng quan trọng.
* **Polynomial Regression & Hiện tượng Overfitting:**
  * Thêm các bậc lũy thừa ($x^2, x^3...$) để học quan hệ phi tuyến.
  * **Đánh đổi Bias - Variance:** Bậc quá thấp gây Underfitting (Bias cao), bậc quá cao gây Overfitting (Variance cao). Lượng dữ liệu lớn là cách chống overfit hiệu quả nhất.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Cài đặt Linear Regression từ scratch**
  * Tự viết class `MyLinearRegression` dùng công thức Normal Equation $(X^T X)^{-1} X^T y$.
  * Kết quả kiểm thử trên bộ dữ liệu California Housing cho hệ số và điểm $R^2$ trùng khớp $100\%$ so với thư viện `sklearn`.

* **Bài 2: Dự đoán giá nhà từ diện tích**
  * Sinh dữ liệu giả lập $200$ căn nhà với hàm gốc: $\text{Giá} = 30 \times \text{Diện tích} + 50 + \text{nhiễu}$.
  * Mô hình tìm lại được hệ số sát với thực tế ($\hat{w} \approx 30.1, \hat{b} \approx 42.9$), $R^2 \approx 0.957$ và trực quan hóa bằng đồ thị phân tán.

* **Bài 3: Khảo sát ảnh hưởng của Outlier**
  * Thêm 5 điểm ngoại lai cực đoan ($y = 100$) vào tập dữ liệu.
  * So sánh cho thấy OLS (Linear Regression) bị kéo lệch hoàn toàn do MSE phạt bậc hai, trong khi `HuberRegressor` vẫn giữ vững kết quả gần sát với đường gốc $(w=3, b=5)$.

* **Bài 4: Chọn lọc đặc trưng với Ridge và Lasso**
  * Thử nghiệm trên dữ liệu 20 đặc trưng (chỉ có 5 đặc trưng thật, 15 đặc trưng là nhiễu).
  * Trong khi OLS và Ridge giữ lại toàn bộ 20 đặc trưng, Lasso ($\alpha=0.1$) đã triệt tiêu chính xác 15 đặc trưng nhiễu về đúng 0 và giữ lại đúng 5 đặc trưng có ý nghĩa.

* **Bài 5: Bậc đa thức và Overfitting**
  * Huấn luyện đa thức từ bậc 1 đến bậc 17 trên dữ liệu hàm sin ($22$ mẫu train).
  * Đường Test RMSE đạt đáy tại **bậc 3** (mô hình tối ưu nhất) và tăng vọt ở các bậc cao do overfit.
  * Khi tăng tập train lên **500 mẫu**, hiện tượng overfit ở bậc cao biến mất hoàn toàn, chứng minh dữ liệu lớn giúp khống chế variance hiệu quả.

---

