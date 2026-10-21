# Bài 05: Cây quyết định & Rừng ngẫu nhiên (Decision Tree & Random Forest)

## 1. Tóm tắt lý thuyết

* **Cây quyết định (Decision Tree):**
  * Mô hình phi tham số phân chia không gian đặc trưng bằng chuỗi các câu hỏi nhị phân (dạng bậc thang).
  * Mô hình có tính minh bạch cao, dễ diễn giải logic và **không nhạy cảm với thang đo của đặc trưng** (không bắt buộc chuẩn hóa).
  * Hạn chế: Cây đơn lẻ rất dễ bị Overfitting nếu để phát triển tự do mà không cắt tỉa (pruning).
* **Tiêu chuẩn phân chia:**
  * **Chỉ số Gini:** $G = 1 - \sum_{i=1}^C p_i^2$, đo độ vẩn đục (tạp chất) của một nút.
  * **Entropy:** $H = -\sum_{i=1}^C p_i \log_2(p_i)$, đo mức độ hỗn loạn thông tin.
  * Thuật toán chọn câu hỏi phân chia tại mỗi bước theo chiến lược tham lam (Greedy Search) để tối đa hóa độ giảm tạp chất ($\Delta G$).
* **Rừng ngẫu nhiên (Random Forest):**
  * Phương pháp Ensemble kết hợp nhiều cây quyết định độc lập thông qua kỹ thuật **Bagging (Bootstrap Aggregating)**: bốc mẫu có hoàn lại (khoảng $63.2\%$ mẫu cho mỗi cây) và chọn ngẫu nhiên tập con đặc trưng ($\approx \sqrt{p}$) tại mỗi nút chia.
  * Dự đoán bằng cách biểu quyết đa số (phân loại) hoặc lấy trung bình (hồi quy), giúp giảm mạnh phương sai (Variance) mà không làm tăng độ chệch (Bias).

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Tính toán chỉ số Gini thủ công**
  * Tự cài đặt hàm tính Gini và độ giảm Gini bằng NumPy.
  * Kết quả tìm được câu hỏi phân chia tối ưu đầu tiên tại ngưỡng `thu_nhap <= 15.05` với $\Delta G = 0.1221$, hoàn toàn trùng khớp với nút gốc của cây tạo bởi Scikit-learn.

* **Bài 2: Khảo sát hiện tượng Overfitting và cắt tỉa cây**
  * Cây phát triển tự do đạt độ sâu 13 tầng với 92 nút lá: Độ chính xác trên tập train đạt tuyệt đối $100\%$, nhưng trên tập test giảm xuống $76.67\%$ do học cả các điểm nhiễu.
  * Áp dụng kiểm định chéo 5-fold tìm được độ sâu tối ưu `max_depth = 5` (27 nút lá): Độ chính xác trên tập test tăng lên **$85.33\%$**, cho thấy vai trò thiết yếu của việc khống chế độ sâu cây.

* **Bài 3: Khảo sát độ nhạy với thang đo đặc trưng**
  * Huấn luyện 3 cây quyết định trên: dữ liệu gốc, dữ liệu chuẩn hóa `StandardScaler`, và dữ liệu nhân cột thu nhập x1000.
  * Cả 3 mô hình đều cho cùng cấu trúc cây, cùng các ngưỡng phân chia tương ứng và độ chính xác kiểm tra bằng nhau tuyệt đối ($0.8533$), chứng minh cây quyết định hoàn toàn bất biến trước các phép biến đổi đơn điệu về thang đo.

* **Bài 4: So sánh tiêu chuẩn Gini vs Entropy và tham số `min_samples_leaf`**
  * Khi đổi tiêu chuẩn sang Entropy: Độ sâu tối ưu vẫn là 5, độ chính xác tập test đạt $84.67\%$ (chênh lệch chưa tới $1\%$ so với Gini).
  * Cắt tỉa bằng `min_samples_leaf`: Giá trị tốt nhất là $10$ (test đạt $84.67\%$), tuy nhiên phương pháp giới hạn `max_depth = 5` vẫn cho kết quả cân bằng tốt hơn ($85.33\%$).

* **Bài 5: Hiệu quả của Rừng ngẫu nhiên và đánh giá đặc trưng**
  * Huấn luyện mô hình Random Forest với 200 cây: Độ chính xác trên tập test tăng vọt lên **$90.00\%$** (vượt trội so với cây đơn lẻ $85.33\%$).
  * So sánh MDI (`feature_importances_`) và Permutation Importance: Cột ngẫu nhiên vô nghĩa bị MDI đánh giá cao giả tạo ($0.150$) do có nhiều giá trị phân biệt, nhưng khi dùng Permutation Importance trên tập test, điểm số của cột này giảm về $0.0017$ (xếp cuối bảng), phản ánh đúng bản chất đặc trưng.

* **Bài 6: Cây và Rừng cho bài toán hồi quy & Khả năng ngoại suy**
  * Huấn luyện hồi quy trên dữ liệu giá nhà: Sai số MAE của Hồi quy tuyến tính ($0.2578$ tỷ) tốt hơn Random Forest ($0.4559$ tỷ) và Decision Tree ($0.5039$ tỷ) do dữ liệu có xu hướng tuyến tính rõ ràng.
  * Khảo sát ngoại suy cho căn nhà $250\text{ m}^2$ (vượt ngoài miền dữ liệu học): Decision Tree dự đoán $9.40$ tỷ và Random Forest dự đoán $9.20$ tỷ (bị giới hạn bởi giá trị trung bình của các mẫu lớn nhất trong tập huấn luyện, không thể ngoại suy tăng tiếp).

---

## 3. Nhận xét tổng kết

* Cây quyết định đơn lẻ phù hợp khi cần một mô hình minh bạch, dễ giải thích logic quyết định và không yêu cầu tiền xử lý dữ liệu phức tạp.
* Rừng ngẫu nhiên là mô hình rất mạnh cho dữ liệu dạng bảng, có độ chính xác cao, ít bị quá khớp khi tăng số lượng cây.
* Cả Cây quyết định và Rừng ngẫu nhiên đều không có khả năng ngoại suy ngoài miền dữ liệu đã học trong bài toán hồi quy.
