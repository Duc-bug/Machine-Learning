# Bài 08: Phân cụm K-Means (K-Means Clustering)

## 1. Tóm tắt lý thuyết

* **Bản chất của phân cụm (Học không giám sát):**
  Dữ liệu không có nhãn đầu ra ($y$), thuật toán tự động phân chia các mẫu dữ liệu thành $K$ nhóm dựa trên mức độ tương đồng về khoảng cách trong không gian đặc trưng.
* **Quy trình hoạt động của K-Means:**
  1. *Khởi tạo:* Chọn ngẫu nhiên $K$ điểm làm tâm cụm ban đầu.
  2. *Gán cụm:* Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất theo khoảng cách Euclid.
  3. *Cập nhật tâm:* Tính lại tọa độ tâm cụm bằng trung bình cộng tọa độ của tất cả các điểm thuộc cụm đó.
  4. Lặp lại bước 2 và 3 cho đến khi tọa độ các tâm cụm không đổi hoặc đạt số vòng lặp tối đa.
* **Hàm mục tiêu Inertia:**
  $$\text{Inertia} = \sum_{j=1}^K \sum_{x \in C_j} \|x - \mu_j\|_2^2$$
  Đo tổng bình phương khoảng cách từ các điểm tới tâm cụm tương ứng. Thuật toán K-Means tìm cách giảm thiểu quán tính này.
* **Phương pháp xác định số cụm $K$ tối ưu:**
  * **Phương pháp khuỷu tay (Elbow Method):** Vẽ đồ thị Inertia theo $K$, chọn điểm $K$ mà tại đó tốc độ giảm Inertia bắt đầu chững lại rõ rệt (tạo thành điểm gập khuỷu tay).
  * **Hệ số Silhouette:** Đo lường mức độ gắn kết bên trong cụm so với khoảng cách tới cụm lân cận gần nhất (giá trị từ $-1$ đến $1$). Điểm $K$ có Silhouette trung bình cao nhất là số cụm phù hợp nhất.
* **Phương pháp khởi tạo `k-means++`:**
  Chọn tâm cụm đầu tiên ngẫu nhiên, các tâm cụm tiếp theo được chọn theo phân phối xác suất tỉ lệ thuận với bình phương khoảng cách tới các tâm đã có, giúp các tâm nằm phân tán đều và giảm nguy cơ kẹt ở cực tiểu địa phương.

---

## 2. Kết quả các bài tập thực hành

* **Bài 1: Tự viết hàm tính Silhouette cho từng điểm**
  * Tự cài đặt công thức tính $a(i)$ (khoảng cách trung bình nội cụm) và $b(i)$ (khoảng cách trung bình tới cụm gần nhất), từ đó tính hệ số Silhouette $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$.
  * Kết quả tính tay hoàn toàn khớp với `sklearn.metrics.silhouette_samples`. Điểm có giá trị Silhouette thấp nhất ($s \approx 0.05$) là điểm nằm ngay tại vùng ranh giới tranh chấp giữa hai cụm khách hàng.

* **Bài 2: Phân cụm khi bổ sung đặc trưng Tuổi**
  * Mở rộng không gian phân khúc khách hàng từ 2 chiều lên 3 chiều (`tuoi`, `thu_nhap`, `chi_tieu`).
  * Sau khi chuẩn hóa bằng `StandardScaler`, khảo sát $K$ từ 2 đến 10 cho thấy $K = 5$ vẫn là số cụm tối ưu nhất với hệ số Silhouette đạt đỉnh **$0.4807$**, chia khách hàng thành 5 nhóm nhân khẩu học và hành vi tiêu dùng rõ rệt.

* **Bài 3: So sánh `k-means++` và khởi tạo ngẫu nhiên**
  * Chạy thử nghiệm 50 lần lặp với `init='random'` và `init='k-means++'` trên cùng một bộ dữ liệu (`n_init=1`).
  * Khởi tạo mặc định `k-means++` đạt được nghiệm tối ưu toàn cục (Inertia = $23.0$) trong **$50/50$ lần chạy ($100\%$)**. Ngược lại, khởi tạo ngẫu nhiên thông thường bị rơi vào cực tiểu địa phương xấu ở 20 trên 50 lần chạy ($40\%$ thất bại).

* **Bài 4: Nén màu ảnh số bằng K-Means**
  * Sử dụng K-Means gom màu trên ảnh gốc 24-bit (khoảng gần 100,000 màu phân biệt) xuống còn bảng màu gồm đúng $K = 16$ màu (chỉ cần 4 bit để mã hóa mỗi pixel).
  * Kết quả: Dung lượng tệp giảm gần 6 lần, trong khi độ sai lệch mức sáng trung bình giữa ảnh nén và ảnh gốc chỉ khoảng $7.97$ mức sáng (mắt thường quan sát gần như không thấy khác biệt). Đã lưu ảnh so sánh tại `anh_goc.png` và `anh_16_mau.png`.

* **Bài 5: Thử nghiệm thuật toán DBSCAN trên dữ liệu hai vầng trăng**
  * Trên tập dữ liệu uốn lượn hai vầng trăng, K-Means phân chia không chính xác do các ranh giới của K-Means chỉ là các đường trung trực thẳng.
  * Thuật toán DBSCAN với tham số $\text{eps} = 0.25$ và $\text{min\_samples} = 5$ phân cụm chính xác theo mật độ, tách biệt hoàn toàn 2 vầng trăng mà không bị giới hạn bởi hình dạng cụm.

* **Bài 6: Phân cụm chữ số viết tay không cần nhãn**
  * Chạy K-Means với $K = 10$ trên tập Digits (64 chiều pixel).
  * Thuật toán gom cụm rất chuẩn xác các chữ số có hình dạng nét đặc trưng và độc lập (như số 0, 4, 6), nhưng xuất hiện sự nhập nhằng giữa các chữ số có hình học tương tự nhau (như số 1 và số 8, hoặc số 3 và số 9).

---

## 3. Nhận xét tổng kết

* K-Means là thuật toán phân cụm nhanh, hiệu quả và có khả năng mở rộng tốt trên các tập dữ liệu lớn; rất phù hợp cho bài toán phân khúc thị trường và nén dữ liệu.
* Chuẩn hóa dữ liệu là yêu cầu bắt buộc trước khi chạy K-Means để tránh việc các đặc trưng có thang đo lớn lấn át khoảng cách Euclid.
* Hạn chế: K-Means chỉ hoạt động tốt khi các cụm có hình dạng cầu lồi (convex) và kích thước tương đối đồng đều; đối với các cụm phi tuyến phức tạp nên cân nhắc DBSCAN hoặc Spectral Clustering.
