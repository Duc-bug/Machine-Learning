# Machine Learning Coursework

Repository tổng hợp các bài tập thực hành môn Học máy và Ứng dụng, triển khai bằng Python sử dụng các thư viện Scikit-learn, NumPy, Pandas và Matplotlib.

---

## Danh mục các bài thực hành

| Bài | Tên bài | Nội dung và thuật toán trọng tâm |
| :--- | :--- | :--- |
| [Lab 1](./Lab%201%20Linear%20regression) | Hồi quy tuyến tính | Linear Regression, Normal Equation, Gradient Descent, Ridge & Lasso, Polynomial Regression. |
| [Lab 2](./Lab%202) | Hồi quy Logistic | Logistic Regression, hàm Sigmoid, Binary Cross-Entropy, ma trận nhầm lẫn, điều chỉnh ngưỡng. |
| [Lab 3](./Lab%203) | Naive Bayes | Định lý Bayes, GaussianNB, MultinomialNB, Laplace smoothing, phân loại văn bản và tin rác. |
| [Lab 4](./Lab%204) | K-Láng giềng gần nhất (KNN) | K-Nearest Neighbors, khoảng cách Euclid và Manhattan, chuẩn hóa đặc trưng, KNN hồi quy. |
| [Lab 5](./Lab%205) | Cây quyết định & Rừng ngẫu nhiên | Decision Tree, Random Forest, chỉ số Gini / Entropy, cắt tỉa cây, Permutation Importance. |
| [Lab 6](./Lab%206) | Máy vectơ hỗ trợ (SVM) | Support Vector Machine, Maximum Margin, tham số phạt C, Kernel Trick (Linear, Poly, RBF). |
| [Lab 7](./Lab%207) | Đánh giá mô hình & Tinh chỉnh | Accuracy, Precision, Recall, F1-score, đường cong ROC-AUC, PR-Curve, xử lý mất cân bằng lớp. |
| [Lab 8](./Lab%208) | Phân cụm K-Means | Phân cụm không giám sát, phương pháp Elbow, hệ số Silhouette, nén màu ảnh, DBSCAN. |
| [Lab 9](./Lab%209) | Mạng nơ-ron nhân tạo (MLP) | Multi-Layer Perceptron, hàm kích hoạt (ReLU, Sigmoid, Tanh), lan truyền thuận/ngược, Early Stopping. |

---

## Cài đặt và sử dụng

Yêu cầu môi trường Python 3.10 trở lên.

```bash
# Clone repository
git clone https://github.com/Duc-bug/Machine-Learning.git
cd Machine-Learning

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### Chạy bài tập
Mỗi thư mục `Lab X` chứa:
- File Jupyter Notebook (`.ipynb`): Mã nguồn thực thi, biểu đồ và kết quả chạy chi tiết.
- Thư mục `data/`: Dữ liệu bài tập.
- File `README.md`: Tóm tắt lý thuyết, kết quả từng bài tập thực hành và nhận xét.

Khởi chạy Jupyter Notebook:
```bash
jupyter notebook
```
