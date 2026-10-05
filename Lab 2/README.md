# Bài 02: Hồi quy Logistic (Logistic Regression)

## 1. Kiến thức cốt lõi học được

* **Bản chất bài toán:**
  * Dùng cho **phân loại nhị phân** ($y \in \{0, 1\}$).
  * Không dùng Hồi quy tuyến tính vì: đầu ra không bị giới hạn trong khoảng xác suất $[0, 1]$, cực kỳ nhạy cảm với ngoại lai (outliers), và sai giả định về phân phối xác suất.
* **Hàm Sigmoid (Logistic Function):**
  $$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{với } z = w^T x + b$$
  * Nén toàn bộ giá trị thực $(-\infty, +\infty)$ về khoảng xác suất $(0, 1)$.
  * Điểm gốc đối xứng: $\sigma(0) = 0.5$ (mốc phân vân $50/50$). Đạo hàm: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
* **Biên quyết định (Decision Boundary):**
  * Là đường/siêu phẳng thỏa mãn $w^T x + b = 0$ (ứng với xác suất $\hat{p} = 0.5$).
  * Với mô hình 1 biến: điểm phân loại nằm tại $x = -\frac{b}{w}$.
* **Hàm mất mát Binary Cross-Entropy (BCE / Log Loss):**
  * Xuất phát từ ước lượng hợp lý cực đại (MLE) của phân phối Bernoulli.
  * Là hàm lồi (convex), đảm bảo Gradient Descent luôn tìm được cực tiểu toàn cục.
* **Ma trận nhầm lẫn (Confusion Matrix) & 4 thước đo:**
  * **Accuracy:** $\frac{TP + TN}{\text{Tổng}}$ — Tỷ lệ đoán đúng tổng thể (dễ sai lệch khi dữ liệu lệch lớp).
  * **Precision:** $\frac{TP}{TP + FP}$ — Đoán Dương thì chuẩn bao nhiêu % (tránh báo động giả / FP).
  * **Recall:** $\frac{TP}{TP + FN}$ — Tìm ra được bao nhiêu % ca Dương thực tế (tránh bỏ sót / FN).
  * **$F_1$-score:** Trung bình điều hòa giữa Precision và Recall: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
* **Ngưỡng quyết định (Decision Threshold) & Sự đánh đổi:**
  * Mặc định là $0.5$, nhưng có thể linh hoạt điều chỉnh theo nghiệp vụ:
  * **Hạ ngưỡng:** Dự đoán dễ dãi hơn $\implies$ Recall tăng, Precision giảm.
  * **Nâng ngưỡng:** Dự đoán khắt khe hơn $\implies$ Precision tăng, Recall giảm.

---

## 2. Bài học rút ra từ thực hành

* **Ranh giới phân vân 50/50 ($x = -b/w$):**
  * Với bài toán dự đoán qua môn theo giờ ôn: $w \approx 0.393$, $b \approx -5.065 \implies x = \frac{5.065}{0.393} \approx 12.89$ giờ. Ôn tập từ $13$ giờ trở lên thì xác suất qua môn $> 50\%$.
* **Mở rộng mô hình nhiều biến:**
  * Thêm đặc trưng *điểm giữa kỳ* giúp Accuracy tăng từ $86.67\%$ lên $90.00\%$. Bổ sung biến có độ phân hóa tốt giúp ranh giới quyết định phân tách dữ liệu chính xác hơn.
* **Ngưỡng tối ưu không nhất thiết là 0.5:**
  * Quét ngưỡng từ $0.05 \to 0.95$ cho thấy ngưỡng $t = 0.60$ đạt $F_1$ cao nhất ($0.9412$ so với $0.8947$ ở $t = 0.5$) nhờ triệt tiêu hoàn toàn các ca báo động giả ($FP = 0$, Precision đạt $100\%$).
* **Ý nghĩa việc chọn "Lớp dương" (Positive Class):**
  * Đổi lớp quan tâm từ Qua môn (1) sang Rớt môn (0) khiến Recall giảm từ $94.44\%$ xuống $75.00\%$. Thước đo luôn phải gắn liền với bài toán thực tế (cần ưu tiên phát hiện sớm sinh viên có nguy cơ rớt môn hay đảm bảo chắc chắn ai sẽ qua môn).
