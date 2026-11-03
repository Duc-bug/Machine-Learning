## 1. Bản chất của Cây quyết định và Rừng ngẫu nhiên
* **Cây quyết định (Decision Tree):** Ra quyết định bằng chuỗi câu hỏi có/không theo kiểu chia đôi không gian (dạng bậc thang). Mô hình rất dễ giải thích (đọc được như luật ngân hàng), chạy nhanh và **không cần chuẩn hoá dữ liệu**. Tuy nhiên, cây đơn lẻ rất dễ **học vẹt (overfitting)** nếu để mọc tự do.
* **Rừng ngẫu nhiên (Random Forest):** Trồng nhiều cây độc lập bằng cách bốc mẫu có hoàn lại (Bootstrap ~63.2% mẫu) và chọn ngẫu nhiên tập con đặc trưng tại mỗi nút ($\sqrt{p}$). Khi dự đoán, các cây bỏ phiếu đa số giúp triệt tiêu sai số cục bộ, giảm phương sai mạnh mẽ và tăng độ chính xác vượt trội.

---

## 2. Nhận xét nhanh từng phần & bài tập

* **Tự tính Gini & câu hỏi gốc (Bài 1 & Bài tập 2):**
  * Tự code tính Gini bằng NumPy cho kết quả giảm Gini tối đa tại ngưỡng `thu_nhap <= 15.05` ($\Delta G = 0.1221$), khớp hoàn toàn với câu hỏi đầu tiên của scikit-learn.
  * Cây luôn chọn câu hỏi theo cách **tham lam (greedy)**: chọn cái tốt nhất ngay tại bước hiện tại mà không nhìn trước các bước sau.

* **Cây học vẹt và kiểm định chéo (Bài 3 & Bài 4):**
  * *Cây tự do:* Đạt sâu 13 tầng, 92 lá, học thuộc lòng tập train ($1.0000$) nhưng tập test rớt xuống $0.7667$ (bị quá khớp, cố ôm lấy các điểm nhiễu).
  * *Kiểm định chéo (5-fold CV):* Chọn được độ sâu tối ưu `max_depth = 5` (27 lá), accuracy trên tập test tăng vọt lên **$0.8533$**.
  * Quy tắc: Ở KNN $k$ nhỏ là học vẹt, còn ở Cây thì độ sâu lớn là học vẹt.

* **Cây không sợ thang đo (Bài 5):**
  * Thử trên dữ liệu gốc, chuẩn hoá `StandardScaler`, và nhân cột thu nhập x1000: cả 3 cây đều cho cùng accuracy **$0.8533$** và $0$ khách bị đoán khác. Khác hoàn toàn với KNN, cây quyết định chỉ so từng cột với một ngưỡng nên không cần chuẩn hoá.

* **Gini vs Entropy & Các cách cắt tỉa (Bài tập 3 & Bài tập 4):**
  * *Đổi sang Entropy:* Cùng chọn độ sâu tối ưu là 5, test accuracy đạt **$0.8467$** (chênh lệch chưa tới 1% so với Gini). Đổi thước đo hầu như không làm kết quả khác biệt.
  * *Cắt tỉa bằng `min_samples_leaf`:* Chọn được giá trị tốt nhất là $10$ (test đạt **$0.8467$**), nhưng vẫn kém hơn cách cắt tỉa bằng `max_depth = 5` (**$0.8533$**).

* **Sức mạnh của Rừng ngẫu nhiên (Bài 6):**
  * Rừng 200 cây kéo accuracy tập test lên tới **$0.9000$** (ăn đứt cây đơn lẻ $0.8533$).
  * Thêm cây không làm rừng bị học vẹt, chỉ làm kết quả ổn định hơn và chạy chậm hơn (có thể tăng tốc bằng `n_jobs=-1`).

* **Cạm bẫy độ quan trọng đặc trưng: MDI vs Permutation (Bài 7 & Bài tập 5):**
  * *Theo MDI (`feature_importances_`):* Cột `so_ngau_nhien` vô nghĩa lại được chấm tới **$0.150$** (xếp hạng 3/5, vượt cả 2 cột thật), do MDI thiên vị các cột liên tục có nhiều giá trị phân biệt.
  * *Theo Permutation Importance (xáo trộn trên test):* Cột ngẫu nhiên tụt ngay xuống **hạng chót (5/5)** với điểm gần bằng 0 ($0.0017$). Khi đánh giá đặc trưng, Permutation Importance trên tập test đáng tin hơn MDI rất nhiều.

* **Cây và Rừng cho bài toán hồi quy (Bài 8 & Bài tập 6):**
  * *Sai số MAE:* Đường thẳng ($0.2578$ tỷ) tốt hơn hẳn Rừng 200 cây ($0.4559$ tỷ) và Cây sâu 3 ($0.5039$ tỷ), vì dữ liệu giá nhà có xu hướng tuyến tính rõ rệt.
  * *Khả năng ngoại suy:* Cả Cây và Rừng **hoàn toàn không ngoại suy được**. Với căn nhà rộng $250\text{ m}^2$ (vượt ngoài tập học), Cây đoán $9.40$ tỷ còn Rừng đoán $9.20$ tỷ (bị kịch trần ở mức giá cao nhất từng thấy trong tập train, thua xa đường thẳng đoán $19.91$ tỷ).

---

## 3. Tóm lại: Khi nào nên / không nên dùng

* **Nên dùng Cây quyết định:** Khi cần giải thích rõ ràng từng bước ra quyết định cho con người (quy trình duyệt vay, y tế), hoặc dữ liệu có nhiều biến phân loại, thang đo lung tung mà lười tiền xử lý.
* **Nên dùng Rừng ngẫu nhiên:** Khi làm việc với dữ liệu dạng bảng (tabular data), cần độ chính xác cao, mô hình khỏe, ít phải tinh chỉnh tham số và không bị quá khớp khi tăng số lượng cây.
* **Không nên dùng (cả Cây lẫn Rừng):** Khi bài toán có xu hướng ngoại suy mạnh ra ngoài miền dữ liệu đã biết (như dự báo chuỗi thời gian tăng trưởng), hoặc bài toán hồi quy có bản chất tuyến tính trơn tru.

