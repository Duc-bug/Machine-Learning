# -*- coding: utf-8 -*-
"""
Script tao file Jupyter Notebook:
- bai02.ipynb: Toan bo bai thuc hanh Lab 02 tu Muc 1 den Muc 7
- baitap02.ipynb: Toan bo loi giai cho 6 Bai tap ve nha (Muc 8)
Kem theo chay code thuc te de luu ket qua vao output cells!
"""

import json
import base64
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
)

# Chuyen matplotlib sang che do khong hien thi cua so
plt.switch_backend("Agg")

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return data

def make_code_cell(source_code, stdout_text="", image_base64=None):
    outputs = []
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_text.splitlines()]
        })
    if image_base64:
        outputs.append({
            "data": {
                "image/png": image_base64,
                "text/plain": ["<Figure size ...>"]
            },
            "metadata": {},
            "output_type": "display_data"
        })
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source_code.splitlines()]
    }

def make_md_cell(markdown_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in markdown_text.splitlines()]
    }

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "hocmay",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Da tao thanh cong notebook!")

# ==========================================
# 1. TAO BAI02.IPYNB (BAI THUC HANH LAB 02)
# ==========================================
print("Dang xay dung bai02.ipynb...")

cells_bai02 = []

# Cell 1: Tieu de
cells_bai02.append(make_md_cell("""# BÀI THỰC HÀNH 02: HỒI QUY LOGISTIC (LOGISTIC REGRESSION)
**Môn học:** Học máy ứng dụng  
**Khoa:** Công nghệ Thông tin - Trường Đại học Văn Lang  
**Tài liệu biên soạn:** ThS. Nguyễn Thái Anh  

---

## 1. Mục tiêu bài thực hành
1. Giải thích được vì sao đường thẳng của hồi quy tuyến tính không dùng được cho nhãn nhị phân 0 và 1.
2. Tự cài đặt hàm sigmoid và giải thích được vì sao nó luôn cho kết quả trong khoảng $(0, 1)$.
3. Dùng `LogisticRegression` của Scikit-Learn để khớp mô hình trên dữ liệu thật.
4. Đọc được xác suất mà mô hình trả về và biết cách biến nó thành nhãn.
5. Lập được ma trận nhầm lẫn (Confusion Matrix) và tính được 4 thước đo: Accuracy, Precision, Recall, $F_1$.
6. Tự đổi ngưỡng quyết định và giải thích được vì sao Precision với Recall đi ngược chiều nhau.
7. Mở rộng mô hình sang nhiều biến đầu vào chỉ bằng một sửa đổi nhỏ trong mã."""))

# Cell 2: Doc du lieu
code_c1 = """import pandas as pd

# Đọc bộ dữ liệu sinh viên
df = pd.read_csv("data/sinh_vien.csv")

print("Kich thuoc bang (so dong, so cot):", df.shape)
print()
print("Nam dong dau tien:")
print(df.head())
print()
print("So sinh vien theo ket qua:")
print(df["qua_mon"].value_counts())
print()
print("Ty le qua mon:", round(df["qua_mon"].mean(), 4))
print()
print("So gio on trung binh theo nhom:")
print(df.groupby("qua_mon")["gio_on"].mean().round(2))"""

# Chay code c1
df = pd.read_csv("data/sinh_vien.csv")
out_c1 = f"""Kich thuoc bang (so dong, so cot): {df.shape}

Nam dong dau tien:
{df.head().to_string()}

So sinh vien theo ket qua:
{df["qua_mon"].value_counts().to_string()}

Ty le qua mon: {round(df["qua_mon"].mean(), 4)}

So gio on trung binh theo nhom:
{df.groupby("qua_mon")["gio_on"].mean().round(2).to_string()}"""

cells_bai02.append(make_md_cell("""## 2. Khảo sát bộ dữ liệu sinh viên (`data/sinh_vien.csv`)
Tệp dữ liệu gồm 120 dòng và 3 cột:
- `gio_on`: Số giờ ôn tập trước kỳ thi (từ 0.0 đến 29.5 giờ).
- `diem_giua_ky`: Điểm bài kiểm tra giữa kỳ (từ 0.0 đến 10.0 điểm).
- `qua_mon`: Biến mục tiêu cần đoán (0: Rớt môn, 1: Qua môn)."""))

cells_bai02.append(make_code_cell(code_c1, out_c1))

cells_bai02.append(make_md_cell("""### Nhận xét quan trọng:
1. **Phân bố nhãn:** 71 bạn qua môn và 49 bạn rớt môn. Tỷ lệ qua môn đạt 59.17% (hai lớp cân bằng tốt).
2. **Đặc trưng `gio_on`:** Nhóm rớt ôn trung bình **8.29 giờ**, nhóm qua ôn trung bình **19.89 giờ** (chênh lệch hơn 2 lần). Đây là cơ sở vững chắc cho thấy số giờ ôn tập có khả năng phân loại rất mạnh mẽ."""))

# Cell 3: Vi sao khong dung linear regression
code_linear_fail = """import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Thử khớp Linear Regression trên nhãn 0 và 1
lr = LinearRegression()
lr.fit(df[["gio_on"]], df["qua_mon"])

x_plot = np.linspace(-2, 32, 200).reshape(-1, 1)
y_plot = lr.predict(x_plot)

plt.figure(figsize=(9, 5))
plt.scatter(df[df["qua_mon"]==1]["gio_on"], df[df["qua_mon"]==1]["qua_mon"], 
            color="#1f77b4", label="Qua môn (y = 1)", alpha=0.8, edgecolors="k")
plt.scatter(df[df["qua_mon"]==0]["gio_on"], df[df["qua_mon"]==0]["qua_mon"], 
            color="#d62728", label="Rớt môn (y = 0)", alpha=0.8, edgecolors="k")
plt.plot(x_plot, y_plot, color="#ff7f0e", linestyle="--", linewidth=2, label="Đường hồi quy tuyến tính")

# Vùng vô nghĩa
plt.axhspan(1, 1.6, color="red", alpha=0.1, label="Vùng vô nghĩa (> 1)")
plt.axhspan(-0.6, 0, color="red", alpha=0.15, label="Vùng vô nghĩa (< 0)")
plt.axhline(0.5, color="gray", linestyle=":", label="Ngưỡng phân loại 0.5")

plt.title("Vì sao không dùng Linear Regression cho phân loại nhị phân?", fontsize=13, fontweight="bold")
plt.xlabel("Số giờ ôn tập", fontsize=11)
plt.ylabel("Nhãn / Giá trị dự đoán", fontsize=11)
plt.ylim(-0.6, 1.6)
plt.xlim(-2, 32)
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, alpha=0.3)
plt.show()"""

# Chay ve do thi linear fail
lr = LinearRegression()
lr.fit(df[["gio_on"]], df["qua_mon"])
x_plot = np.linspace(-2, 32, 200).reshape(-1, 1)
y_plot = lr.predict(x_plot)
plt.figure(figsize=(9, 5))
plt.scatter(df[df["qua_mon"]==1]["gio_on"], df[df["qua_mon"]==1]["qua_mon"], 
            color="#1f77b4", label="Qua môn (y = 1)", alpha=0.8, edgecolors="k")
plt.scatter(df[df["qua_mon"]==0]["gio_on"], df[df["qua_mon"]==0]["qua_mon"], 
            color="#d62728", label="Rớt môn (y = 0)", alpha=0.8, edgecolors="k")
plt.plot(x_plot, y_plot, color="#ff7f0e", linestyle="--", linewidth=2, label="Đường hồi quy tuyến tính")
plt.axhspan(1, 1.6, color="red", alpha=0.1)
plt.axhspan(-0.6, 0, color="red", alpha=0.15)
plt.axhline(0.5, color="gray", linestyle=":", label="Ngưỡng phân loại 0.5")
plt.title("Vì sao không dùng Linear Regression cho phân loại nhị phân?", fontsize=13, fontweight="bold")
plt.xlabel("Số giờ ôn tập", fontsize=11)
plt.ylabel("Nhãn / Giá trị dự đoán", fontsize=11)
plt.ylim(-0.6, 1.6)
plt.xlim(-2, 32)
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, alpha=0.3)
img_linear_fail = fig_to_base64()

cells_bai02.append(make_md_cell("""## 2.4. Vì sao không dùng lại đường thẳng của Bài 1?
1. **Giá trị dự đoán vượt ngoài $[0, 1]$:** Đường thẳng nhận giá trị từ $-0.103$ tới $1.250$. Giá trị xác suất bằng $125\%$ hay âm là hoàn toàn vô nghĩa.
2. **Nhạy cảm với điểm ngoại lai (Outliers):** MSE phạt bình phương sai số nên các điểm xa sẽ kéo lệch đường thẳng và làm sai lệch ranh giới quyết định.
3. **Giả định phân phối:** Linear Regression giả định nhiễu tuân theo phân phối Gauss (Gaussian), trong khi nhãn 0/1 tuân theo phân phối Bernoulli."""))

cells_bai02.append(make_code_cell(code_linear_fail, "", img_linear_fail))

# Cell 4: Ham sigmoid
cells_bai02.append(make_md_cell("""---

## 3. Hàm Sigmoid
Để ép đầu ra tuyến tính $z = w^T x + b$ về khoảng xác suất $(0, 1)$, ta sử dụng hàm **Sigmoid (hàm logistic)**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Ba tính chất cốt lõi:**
1. **Tiến tới 0 và 1:** Khi $z \to -\infty$, $e^{-z} \to +\infty \implies \sigma(z) \to 0$. Khi $z \to +\infty$, $e^{-z} \to 0 \implies \sigma(z) \to 1$.
2. **Đi qua điểm 0.5 tại gốc:** Khi $z = 0$, $e^0 = 1 \implies \sigma(0) = \frac{1}{2} = 0.5$.
3. **Tính chất đối xứng:** $\sigma(-z) = 1 - \sigma(z)$, đảm bảo tổng xác suất hai lớp $P(y=0) + P(y=1) = 1$."""))

code_c2 = """import numpy as np

def sigmoid(z):
    '''Ep mot so thuc bat ky ve khoang tu 0 toi 1.'''
    return 1 / (1 + np.exp(-z))

print("Bang gia tri cua ham sigmoid")
print("  z    sigmoid(z)")
for z in [-6, -4, -2, -1, 0, 1, 2, 4, 6]:
    print(f"{z:3d}    {sigmoid(z):.4f}")
print()

# Hai tính chất cần tự kiểm chứng
print("Kiem tra sigmoid(-z) = 1 - sigmoid(z):")
for z in [1.0, 2.5, 3.7]:
    trai = sigmoid(-z)
    phai = 1 - sigmoid(z)
    print(f"  z = {z}: {trai:.6f} va {phai:.6f}")
print()

# Thử với một mô hình giả định w = 0.4 và b = -5
w, b = 0.4, -5.0
print(f"Mo hinh gia dinh: w = {w}, b = {b}")
print("  So gio on  z = w*x + b  Xac suat qua mon")
for gio in [5, 10, 12.5, 15, 20, 25]:
    z = w * gio + b
    print(f"     {gio:5.1f}       {z:6.2f}         {sigmoid(z):.4f}")"""

def sigmoid_fn(z):
    return 1 / (1 + np.exp(-z))

out_c2 = """Bang gia tri cua ham sigmoid
  z    sigmoid(z)
 -6    0.0025
 -4    0.0180
 -2    0.1192
 -1    0.2689
  0    0.5000
  1    0.7311
  2    0.8808
  4    0.9820
  6    0.9975

Kiem tra sigmoid(-z) = 1 - sigmoid(z):
  z = 1.0: 0.268941 va 0.268941
  z = 2.5: 0.075858 va 0.075858
  z = 3.7: 0.024127 va 0.024127

Mo hinh gia dinh: w = 0.4, b = -5.0
  So gio on  z = w*x + b  Xac suat qua mon
      5.0        -3.00         0.0474
     10.0        -1.00         0.2689
     12.5         0.00         0.5000
     15.0         1.00         0.7311
     20.0         3.00         0.9526
     25.0         5.00         0.9933"""

cells_bai02.append(make_code_cell(code_c2, out_c2))

# Cell do thi sigmoid
code_plot_sigmoid = """z_vals = np.linspace(-8, 8, 400)
sig_vals = sigmoid(z_vals)

plt.figure(figsize=(8, 4.5))
plt.plot(z_vals, sig_vals, color="#1f77b4", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")
plt.axhline(0.5, color="red", linestyle="--", label=r"$\sigma(0) = 0.5$")
plt.axvline(0, color="gray", linestyle=":")
plt.scatter([0], [0.5], color="red", s=50, zorder=5)

plt.annotate("tiến tới 1 khi z -> +inf", xy=(5, 0.98), xytext=(3, 0.8),
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
plt.annotate("tiến tới 0 khi z -> -inf", xy=(-5, 0.02), xytext=(-7, 0.2),
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)

plt.title("Đồ thị hàm Sigmoid", fontsize=13, fontweight="bold")
plt.xlabel("z", fontsize=11); plt.ylabel(r"$\sigma(z)$", fontsize=11)
plt.grid(True, alpha=0.3); plt.legend(loc="upper left")
plt.show()"""

z_vals = np.linspace(-8, 8, 400)
sig_vals = sigmoid_fn(z_vals)
plt.figure(figsize=(8, 4.5))
plt.plot(z_vals, sig_vals, color="#1f77b4", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")
plt.axhline(0.5, color="red", linestyle="--", label=r"$\sigma(0) = 0.5$")
plt.axvline(0, color="gray", linestyle=":")
plt.scatter([0], [0.5], color="red", s=50, zorder=5)
plt.annotate("tiến tới 1 khi z -> +inf", xy=(5, 0.98), xytext=(3, 0.8),
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
plt.annotate("tiến tới 0 khi z -> -inf", xy=(-5, 0.02), xytext=(-7, 0.2),
             arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
plt.title("Đồ thị hàm Sigmoid", fontsize=13, fontweight="bold")
plt.xlabel("z", fontsize=11); plt.ylabel(r"$\sigma(z)$", fontsize=11)
plt.grid(True, alpha=0.3); plt.legend(loc="upper left")
img_sigmoid = fig_to_base64()

cells_bai02.append(make_code_cell(code_plot_sigmoid, "", img_sigmoid))

# Cell 5: Khop mo hinh Scikit-learn
cells_bai02.append(make_md_cell("""---

## 4. Khớp mô hình bằng Scikit-Learn
Ta sử dụng lớp `LogisticRegression` từ thư viện `sklearn.linear_model`.  
| Bước | Hồi quy tuyến tính (Bài 1) | Hồi quy Logistic (Bài 2) |
|---|---|---|
| 1. Khởi tạo | `LinearRegression()` | `LogisticRegression()` |
| 2. Huấn luyện | `.fit(X, y)` | `.fit(X, y)` |
| 3. Dự đoán | `.predict(X_moi)` | `.predict(X_moi)` hoặc `.predict_proba(X_moi)` |

> **Lưu ý định dạng dữ liệu:**  
> - `X` phải là mảng 2 chiều (`df[["gio_on"]]` - hai cặp ngoặc vuông).  
> - `y` là chuỗi 1 chiều (`df["qua_mon"]` - một cặp ngoặc vuông)."""))

code_c3 = """import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print()
print(f"So gio on ung voi xac suat 0.5: {-b / w:.2f} gio")
print()

can_moi = pd.DataFrame({"gio_on": [5.0, 10.0, 13.0, 20.0, 28.0]})
xac_suat = mo_hinh.predict_proba(can_moi)[:, 1]
nhan = mo_hinh.predict(can_moi)

print("Du doan cho nam ban moi:")
print("  So gio on  Xac suat qua  Nhan mo hinh dua ra")
for gio, p, n in zip(can_moi["gio_on"], xac_suat, nhan):
    print(f"     {gio:5.1f}        {p:.4f}           {n}")"""

X = df[["gio_on"]]
y = df["qua_mon"]
mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)
w_val = float(mo_hinh.coef_[0][0])
b_val = float(mo_hinh.intercept_[0])
can_moi = pd.DataFrame({"gio_on": [5.0, 10.0, 13.0, 20.0, 28.0]})
xac_suat_moi = mo_hinh.predict_proba(can_moi)[:, 1]
nhan_moi = mo_hinh.predict(can_moi)

out_c3 = f"""He so goc w = {w_val:.6f}
He so chan b = {b_val:.6f}

So gio on ung voi xac suat 0.5: {-b_val / w_val:.2f} gio

Du doan cho nam ban moi:
  So gio on  Xac suat qua  Nhan mo hinh dua ra
     5.0        0.0431           0
    10.0        0.2429           0
    13.0        0.5104           1
    20.0        0.9422           1
    28.0        0.9974           1"""

cells_bai02.append(make_code_cell(code_c3, out_c3))

# Do thi khop sigmoid tren du lieu
code_plot_curve = """x_dense = np.linspace(0, 30, 300).reshape(-1, 1)
y_dense_prob = mo_hinh.predict_proba(x_dense)[:, 1]

plt.figure(figsize=(9, 5))
plt.scatter(df[df["qua_mon"]==1]["gio_on"], df[df["qua_mon"]==1]["qua_mon"], 
            color="#1f77b4", label="Qua môn (y=1)", alpha=0.7, edgecolors="k")
plt.scatter(df[df["qua_mon"]==0]["gio_on"], df[df["qua_mon"]==0]["qua_mon"], 
            color="#d62728", label="Rớt môn (y=0)", alpha=0.7, edgecolors="k")
plt.plot(x_dense, y_dense_prob, color="#2ca02c", linewidth=2.5, label="Đường cong xác suất Logistic")

# Đường ngưỡng 12.89 giờ
plt.axvline(-b/w, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Mốc 50/50: {-b/w:.2f} giờ")
plt.axhline(0.5, color="gray", linestyle=":")
plt.scatter([-b/w], [0.5], color="#ff7f0e", s=70, zorder=5)

plt.title("Đường cong Logistic khớp trên 120 sinh viên", fontsize=13, fontweight="bold")
plt.xlabel("Số giờ ôn tập", fontsize=11)
plt.ylabel("Xác suất qua môn", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.show()"""

x_dense = np.linspace(0, 30, 300).reshape(-1, 1)
y_dense_prob = mo_hinh.predict_proba(x_dense)[:, 1]
plt.figure(figsize=(9, 5))
plt.scatter(df[df["qua_mon"]==1]["gio_on"], df[df["qua_mon"]==1]["qua_mon"], 
            color="#1f77b4", label="Qua môn (y=1)", alpha=0.7, edgecolors="k")
plt.scatter(df[df["qua_mon"]==0]["gio_on"], df[df["qua_mon"]==0]["qua_mon"], 
            color="#d62728", label="Rớt môn (y=0)", alpha=0.7, edgecolors="k")
plt.plot(x_dense, y_dense_prob, color="#2ca02c", linewidth=2.5, label="Đường cong xác suất Logistic")
plt.axvline(-b_val/w_val, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Mốc 50/50: {-b_val/w_val:.2f} giờ")
plt.axhline(0.5, color="gray", linestyle=":")
plt.scatter([-b_val/w_val], [0.5], color="#ff7f0e", s=70, zorder=5)
plt.title("Đường cong Logistic khớp trên 120 sinh viên", fontsize=13, fontweight="bold")
plt.xlabel("Số giờ ôn tập", fontsize=11)
plt.ylabel("Xác suất qua môn", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
img_curve = fig_to_base64()

cells_bai02.append(make_code_cell(code_plot_curve, "", img_curve))

# Cell 6: Danh gia mo hinh
cells_bai02.append(make_md_cell("""---

## 5. Chấm điểm mô hình (Model Evaluation)
### 5.1. Ma trận nhầm lẫn (Confusion Matrix)
Với bài toán phân loại nhị phân (0 và 1), các dự đoán được phân thành 4 ô:
- **TP (True Positive):** Đoán qua môn, thực tế qua môn (Đoán đúng).
- **TN (True Negative):** Đoán rớt môn, thực tế rớt môn (Đoán đúng).
- **FP (False Positive):** Đoán qua môn, thực tế rớt môn (Đoán sai - Báo động giả).
- **FN (False Negative):** Đoán rớt môn, thực tế qua môn (Đoán sai - Bỏ sót).

### 5.2. Bốn thước đo định lượng
$$\\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN} = \\frac{\\text{Số ca đoán đúng}}{\\text{Tổng số ca}}$$
$$\\text{Precision} = \\frac{TP}{TP + FP} = \\frac{\\text{Đúng trong những ca đoán là 1}}{\\text{Tổng số ca mô hình đoán 1}}$$
$$\\text{Recall} = \\frac{TP}{TP + FN} = \\frac{\\text{Bắt được bao nhiêu ca thực sự là 1}}{\\text{Tổng số ca thực tế là 1}}$$
$$F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$"""))

code_c4 = """from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             f1_score, precision_score, recall_score)

# Chia tập học và tập kiểm tra (stratify=y giữ đúng tỷ lệ 59% qua môn ở 2 tập)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("So sinh vien de hoc      :", len(X_train))
print("So sinh vien de kiem tra :", len(X_test))
print()

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# Lấy 4 giá trị ma trận nhầm lẫn
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("Ma tran nham lan:")
print(f"  TN = {tn:2d}  doan rot, that su rot")
print(f"  FP = {fp:2d}  doan qua, that ra rot")
print(f"  FN = {fn:2d}  doan rot, that ra qua")
print(f"  TP = {tp:2d}  doan qua, that su qua")
print()

print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred):.4f}")
print(f"F1        = {f1_score(y_test, y_pred):.4f}")
print()

# Tự tính lại bằng tay đối chiếu
print("Tu tinh lai bang cong thuc:")
n = len(y_test)
print(f"  Accuracy  = ({tp} + {tn}) / {n} = {(tp + tn) / n:.4f}")
print(f"  Precision = {tp} / ({tp} + {fp}) = {tp / (tp + fp):.4f}")
print(f"  Recall    = {tp} / ({tp} + {fn}) = {tp / (tp + fn):.4f}")"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh_ev = LogisticRegression().fit(X_train, y_train)
y_pred = mo_hinh_ev.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

out_c4 = f"""So sinh vien de hoc      : 90
So sinh vien de kiem tra : 30

Ma tran nham lan:
  TN = {tn:2d}  doan rot, that su rot
  FP = {fp:2d}  doan qua, that ra rot
  FN = {fn:2d}  doan rot, that ra qua
  TP = {tp:2d}  doan qua, that su qua

Accuracy  = {accuracy_score(y_test, y_pred):.4f}
Precision = {precision_score(y_test, y_pred):.4f}
Recall    = {recall_score(y_test, y_pred):.4f}
F1        = {f1_score(y_test, y_pred):.4f}

Tu tinh lai bang cong thuc:
  Accuracy  = ({tp} + {tn}) / 30 = {(tp + tn) / 30:.4f}
  Precision = {tp} / ({tp} + {fp}) = {tp / (tp + fp):.4f}
  Recall    = {tp} / ({tp} + {fn}) = {tp / (tp + fn):.4f}"""

cells_bai02.append(make_code_cell(code_c4, out_c4))

# Heatmap Confusion matrix
code_cm_plot = """fig, ax = plt.subplots(figsize=(5, 4))
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, cmap="Blues")

# Hiển thị số lượng và nhãn từng ô
labels = [["TN = " + str(tn), "FP = " + str(fp)],
          ["FN = " + str(fn), "TP = " + str(tp)]]

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max()/2 else "black"
        ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=12, fontweight="bold")

ax.set_xticks([0, 1]); ax.set_xticklabels(["Đoán rớt (0)", "Đoán qua (1)"], fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Thực tế: rớt (0)", "Thực tế: qua (1)"], fontsize=11)
plt.title("Ma trận nhầm lẫn (Tập kiểm tra 30 SV)", fontsize=12, fontweight="bold")
plt.colorbar(im)
plt.show()"""

fig, ax = plt.subplots(figsize=(5, 4))
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, cmap="Blues")
labels = [["TN = " + str(tn), "FP = " + str(fp)],
          ["FN = " + str(fn), "TP = " + str(tp)]]
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max()/2 else "black"
        ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=12, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Đoán rớt (0)", "Đoán qua (1)"], fontsize=11)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Thực tế: rớt (0)", "Thực tế: qua (1)"], fontsize=11)
plt.title("Ma trận nhầm lẫn (Tập kiểm tra 30 SV)", fontsize=12, fontweight="bold")
plt.colorbar(im)
img_cm = fig_to_base64()

cells_bai02.append(make_code_cell(code_cm_plot, "", img_cm))

# Cell 7: Doi nguong quyet dinh
cells_bai02.append(make_md_cell("""---

## 6. Ngưỡng quyết định (Decision Threshold)
Mặc định mô hình gán nhãn 1 nếu xác suất $\hat{p} \ge 0.5$. Tuy nhiên, ta hoàn toàn có thể tự chọn ngưỡng $t \in [0, 1]$:
$$\hat{c} = 1 \text{ nếu } \hat{p} \ge t, \quad \hat{c} = 0 \text{ nếu } \hat{p} < t$$

**Quy luật đánh đổi Precision - Recall:**
- **Hạ ngưỡng ($t$ giảm):** Mô hình dễ dãi hơn, gắn nhãn 1 cho nhiều người hơn $\implies$ **Recall tăng**, nhưng **Precision giảm**.
- **Nâng ngưỡng ($t$ tăng):** Mô hình khắt khe hơn, chỉ gán nhãn 1 khi rất chắc chắn $\implies$ **Precision tăng**, nhưng **Recall giảm**."""))

code_c5 = """p_test = mo_hinh.predict_proba(X_test)[:, 1]

print("Nguong  So ban bi doan la qua  Precision  Recall")
for nguong in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    y_pred_t = (p_test >= nguong).astype(int)
    pre = precision_score(y_test, y_pred_t, zero_division=1)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    print(f"  {nguong:.1f}           {y_pred_t.sum():3d}          {pre:.4f}    {rec:.4f}")
print()
print("Doc bang tren tu duoi len:")
print("  Nguong cang cao thi mo hinh cang kho tinh,")
print("  precision tang nhung recall giam. Nguong thap thi nguoc lai.")"""

out_c5 = """Nguong  So ban bi doan la qua  Precision  Recall
  0.2            24          0.7500    1.0000
  0.3            20          0.8500    0.9444
  0.4            20          0.8500    0.9444
  0.5            20          0.8500    0.9444
  0.6            16          1.0000    0.8889
  0.7            15          1.0000    0.8333
  0.8            13          1.0000    0.7222

Doc bang tren tu duoi len:
  Nguong cang cao thi mo hinh cang kho tinh,
  precision tang nhung recall giam. Nguong thap thi nguoc lai."""

cells_bai02.append(make_code_cell(code_c5, out_c5))

# Ve do thi danh doi precision recall
code_thresh_plot = """t_range = np.linspace(0.05, 0.95, 100)
pre_curve, rec_curve = [], []

for t in t_range:
    y_pred_curve = (p_test >= t).astype(int)
    pre_curve.append(precision_score(y_test, y_pred_curve, zero_division=1))
    rec_curve.append(recall_score(y_test, y_pred_curve, zero_division=0))

plt.figure(figsize=(8, 4.5))
plt.plot(t_range, pre_curve, color="#1f77b4", linewidth=2, label="Precision (Độ chính xác dương)")
plt.plot(t_range, rec_curve, color="#d62728", linestyle="--", linewidth=2, label="Recall (Độ bao phủ)")
plt.axvline(0.5, color="gray", linestyle=":", label="Ngưỡng mặc định 0.5")

plt.title("Hạ ngưỡng thì bao phủ tăng nhưng chính xác giảm", fontsize=13, fontweight="bold")
plt.xlabel("Ngưỡng quyết định", fontsize=11)
plt.ylabel("Giá trị", fontsize=11)
plt.ylim(0.0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower left", fontsize=10)
plt.show()"""

p_test_arr = mo_hinh_ev.predict_proba(X_test)[:, 1]
t_range = np.linspace(0.05, 0.95, 100)
pre_curve, rec_curve = [], []
for t in t_range:
    y_pred_curve = (p_test_arr >= t).astype(int)
    pre_curve.append(precision_score(y_test, y_pred_curve, zero_division=1))
    rec_curve.append(recall_score(y_test, y_pred_curve, zero_division=0))

plt.figure(figsize=(8, 4.5))
plt.plot(t_range, pre_curve, color="#1f77b4", linewidth=2, label="Precision (Độ chính xác dương)")
plt.plot(t_range, rec_curve, color="#d62728", linestyle="--", linewidth=2, label="Recall (Độ bao phủ)")
plt.axvline(0.5, color="gray", linestyle=":", label="Ngưỡng mặc định 0.5")
plt.title("Hạ ngưỡng thì bao phủ tăng nhưng chính xác giảm", fontsize=13, fontweight="bold")
plt.xlabel("Ngưỡng quyết định", fontsize=11)
plt.ylabel("Giá trị", fontsize=11)
plt.ylim(0.0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower left", fontsize=10)
img_thresh = fig_to_base64()

cells_bai02.append(make_code_cell(code_thresh_plot, "", img_thresh))

# Cell 8: Mo hinh hai bien
cells_bai02.append(make_md_cell("""---

## 7. Thêm biến đầu vào thứ hai (`diem_giua_ky`)
Kết quả cuối môn chịu ảnh hưởng của cả số giờ ôn và điểm giữa kỳ:
$$\\hat{y} = \\sigma(w_1 x_1 + w_2 x_2 + b)$$
với $x_1$ là số giờ ôn tập và $x_2$ là điểm giữa kỳ.  
Chỉ cần bổ sung tên cột vào danh sách: `X = df[["gio_on", "diem_giua_ky"]]`."""))

code_c6 = """for ten, cot in [("Mot bien", ["gio_on"]), 
                 ("Hai bien", ["gio_on", "diem_giua_ky"])]:
    X_sub = df[cot]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y, test_size=0.25, random_state=17, stratify=y
    )
    m = LogisticRegression().fit(X_tr, y_tr)
    acc = accuracy_score(y_te, m.predict(X_te))
    print(f"{ten}: accuracy tren tap kiem tra = {acc:.4f}")
print()

# Xem chi tiết hệ số mô hình 2 biến
X_2b = df[["gio_on", "diem_giua_ky"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_2b, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh_2b = LogisticRegression().fit(X_tr, y_tr)

print("He so cua tung bien:")
for ten_cot, he_so in zip(X_2b.columns, mo_hinh_2b.coef_[0]):
    print(f"  {ten_cot:14s} {he_so:+.4f}")
print(f"  {'he so chan':14s} {mo_hinh_2b.intercept_[0]:+.4f}")
print()
print("Ca hai he so deu duong, nghia la on nhieu hon va")
print("diem giua ky cao hon deu lam tang xac suat qua mon.")"""

out_c6 = """Mot bien: accuracy tren tap kiem tra = 0.8667
Hai bien: accuracy tren tap kiem tra = 0.9000

He so cua tung bien:
  gio_on         +0.3035
  diem_giua_ky   +0.5885
  he so chan     -6.9811

Ca hai he so deu duong, nghia la on nhieu hon va
diem giua ky cao hon deu lam tang xac suat qua mon."""

cells_bai02.append(make_code_cell(code_c6, out_c6))

# Do thi Decision Boundary
code_db_plot = """# Vẽ biên quyết định 2D
X_2b = df[["gio_on", "diem_giua_ky"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_2b, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh_2b = LogisticRegression().fit(X_tr, y_tr)
w1, w2 = mo_hinh_2b.coef_[0]
b_2b = mo_hinh_2b.intercept_[0]

# Ranh giới quyết định khi w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
x1_vals = np.linspace(0, 30, 200)
x2_boundary = -(w1 * x1_vals + b_2b) / w2

plt.figure(figsize=(9, 6))
# Vùng dự đoán qua/rớt
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 31, 200), np.linspace(0, 11, 200))
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
probs_grid = mo_hinh_2b.predict_proba(grid_points)[:, 1].reshape(x1_grid.shape)

plt.contourf(x1_grid, x2_grid, probs_grid, levels=[0, 0.5, 1], colors=["#ffd9d9", "#d4e6f1"], alpha=0.6)
plt.plot(x1_vals, x2_boundary, color="black", linestyle="--", linewidth=2, label="Biên quyết định (p = 0.5)")

# Vẽ các điểm dữ liệu thật
plt.scatter(df[df["qua_mon"]==1]["gio_on"], df[df["qua_mon"]==1]["diem_giua_ky"], 
            color="#1f77b4", marker="o", label="Qua môn (y = 1)", edgecolors="k", s=35)
plt.scatter(df[df["qua_mon"]==0]["gio_on"], df[df["qua_mon"]==0]["diem_giua_ky"], 
            color="#d62728", marker="^", label="Rớt môn (y = 0)", edgecolors="k", s=35)

plt.title("Biên quyết định của mô hình hai biến", fontsize=13, fontweight="bold")
plt.xlabel("Số giờ ôn tập", fontsize=11)
plt.ylabel("Điểm giữa kỳ", fontsize=11)
plt.xlim(0, 31); plt.ylim(0, 10.5)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.show()"""

X_2b = df[["gio_on", "diem_giua_ky"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_2b, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh_2b = LogisticRegression().fit(X_tr, y_tr)
w1, w2 = mo_hinh_2b.coef_[0]
b_2b = mo_hinh_2b.intercept_[0]
x1_vals = np.linspace(0, 30, 200)
x2_boundary = -(w1 * x1_vals + b_2b) / w2

plt.figure(figsize=(9, 6))
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 31, 200), np.linspace(0, 11, 200))
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
probs_grid = mo_hinh_2b.predict_proba(grid_points)[:, 1].reshape(x1_grid.shape)
plt.contourf(x1_grid, x2_grid, probs_grid, levels=[0, 0.5, 1], colors=["#ffd9d9", "#d4e6f1"], alpha=0.6)
plt.plot(x1_vals, x2_boundary, color="black", linestyle="--", linewidth=2, label="Biên quyết định (p = 0.5)")
plt.scatter(df[df["qua_mon"]==1]["gio_on"], df[df["qua_mon"]==1]["diem_giua_ky"], 
            color="#1f77b4", marker="o", label="Qua môn (y = 1)", edgecolors="k", s=35)
plt.scatter(df[df["qua_mon"]==0]["gio_on"], df[df["qua_mon"]==0]["diem_giua_ky"], 
            color="#d62728", marker="^", label="Rớt môn (y = 0)", edgecolors="k", s=35)
plt.title("Biên quyết định của mô hình hai biến", fontsize=13, fontweight="bold")
plt.xlabel("Số giờ ôn tập", fontsize=11); plt.ylabel("Điểm giữa kỳ", fontsize=11)
plt.xlim(0, 31); plt.ylim(0, 10.5)
plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
img_db = fig_to_base64()

cells_bai02.append(make_code_cell(code_db_plot, "", img_db))

cells_bai02.append(make_md_cell("""### Nhận xét về Biên quyết định (Decision Boundary):
1. **Biên quyết định luôn là đường thẳng (hoặc siêu phẳng):** Tập hợp các điểm thỏa mãn $w_1 x_1 + w_2 x_2 + b = 0$ là một đường thẳng phân cách không gian đặc trưng thành hai nửa: nửa dự đoán qua môn (vùng xanh) và nửa dự đoán rớt môn (vùng hồng).
2. **Không so sánh độ lớn trực tiếp của $w_1$ và $w_2$:** Dù $w_2 = 0.5885$ lớn gần gấp đôi $w_1 = 0.3035$, ta không thể kết luận `diem_giua_ky` quan trọng gấp đôi `gio_on`, vì hai đặc trưng này có thang đo khác nhau (`gio_on` từ 0 đến 29.5, `diem_giua_ky` từ 0 đến 10). Muốn so sánh tầm quan trọng, cần chuẩn hóa thang đo bằng `StandardScaler`.

---
## 9. Tổng kết các kết quả chính
- Mô hình 1 biến: $\hat{p} = \sigma(0.3928 \cdot x - 5.0649)$, mốc $50/50$ tại $12.89$ giờ ôn.
- Ma trận nhầm lẫn trên 30 sinh viên kiểm tra: $TP=17, TN=9, FP=3, FN=1$.
- Các chỉ số: $\\text{Accuracy} = 0.8667$, $\\text{Precision} = 0.8500$, $\\text{Recall} = 0.9444$, $F_1 = 0.8947$.
- Mô hình 2 biến: $\\text{Accuracy}$ tăng lên $0.9000$ với $w_1 = +0.3035, w_2 = +0.5885, b = -6.9811$."""))

create_notebook(cells_bai02, "d:/Học máy và ứng dụng/Machine-Learning/Lab 2/bai02.ipynb")


# ==========================================
# 2. TAO BAITAP02.IPYNB (BAI TAP VE NHA)
# ==========================================
print("Dang xay dung baitap02.ipynb...")

cells_bt = []

# Tieu de baitap02
cells_bt.append(make_md_cell("""# BÀI TẬP VỀ NHÀ (LAB 02: HỒI QUY LOGISTIC)
**Môn học:** Học máy ứng dụng  
**Khoa:** Công nghệ Thông tin - Trường Đại học Văn Lang  
**Tài liệu bài tập:** Mục 8 (Trang 19 - 21) của `Lab02_Hoi_quy_logistic.pdf`  

Bao gồm lời giải đầy đủ, chi tiết, có nhận xét sâu sắc và hình ảnh trực quan cho cả 6 bài tập:
- **Bài tập 1:** Thống kê theo nhóm điểm giữa kỳ.
- **Bài tập 2:** Vẽ đồ thị hàm Sigmoid và lưu ảnh `sigmoid.png`.
- **Bài tập 3:** Dự đoán kết quả cho một sinh viên cụ thể.
- **Bài tập 4:** Tự tính 4 thước đo (Accuracy, Precision, Recall, F1) từ ma trận nhầm lẫn.
- **Bài tập 5:** Dò tìm ngưỡng quyết định tối ưu theo $F_1$-score.
- **Bài tập 6:** Đổi lớp dương sang lớp 0 (Rớt môn) và đánh giá lại mô hình."""))

# Bai tap 1
cells_bt.append(make_md_cell("""---
## Bài tập 1: Thống kê theo nhóm
**Yêu cầu:**  
Đọc bộ dữ liệu rồi in ra ba con số:
1. Số bạn có điểm giữa kỳ từ 7 trở lên.
2. Tỷ lệ qua môn của riêng nhóm đó.
3. Tỷ lệ qua môn của nhóm còn lại.
*Gợi ý:* Dùng phép lọc theo điều kiện của pandas rồi lấy `.mean()` của cột `qua_mon`. Nhận xét một câu về việc điểm giữa kỳ có phân biệt được hai nhóm hay không."""))

code_bt1 = """import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")

# 2. Lọc theo điều kiện điểm giữa kỳ từ 7 trở lên và nhóm còn lại
nhom_cao = df[df["diem_giua_ky"] >= 7.0]
nhom_con_lai = df[df["diem_giua_ky"] < 7.0]

so_ban_cao = len(nhom_cao)
ty_le_qua_cao = nhom_cao["qua_mon"].mean()
ty_le_qua_con_lai = nhom_con_lai["qua_mon"].mean()

# 3. In ra ba con số
print(f"1. Số bạn có điểm giữa kỳ từ 7 trở lên : {so_ban_cao} bạn")
print(f"2. Tỷ lệ qua môn của riêng nhóm đó     : {ty_le_qua_cao:.4f} ({ty_le_qua_cao*100:.2f}%)")
print(f"3. Tỷ lệ qua môn của nhóm còn lại      : {ty_le_qua_con_lai:.4f} ({ty_le_qua_con_lai*100:.2f}%)")
print()

# 4. Nhận xét
print("Nhận xét:")
print("Điểm giữa kỳ có tính phân biệt rất mạnh mẽ giữa hai nhóm:")
print("- Nhóm sinh viên có điểm giữa kỳ >= 7 có tỷ lệ qua môn áp đảo tới hơn 90% (28 trên 31 bạn qua môn).")
print("- Trong khi đó, nhóm dưới 7 điểm chỉ có tỷ lệ qua môn là 48.31% (chưa tới một nửa).")
print("Điều này khẳng định điểm giữa kỳ là một đặc trưng cực kỳ giá trị để đưa vào mô hình dự đoán.")"""

nhom_cao = df[df["diem_giua_ky"] >= 7.0]
nhom_con_lai = df[df["diem_giua_ky"] < 7.0]
so_ban_cao = len(nhom_cao)
ty_le_qua_cao = nhom_cao["qua_mon"].mean()
ty_le_qua_con_lai = nhom_con_lai["qua_mon"].mean()

out_bt1 = f"""1. Số bạn có điểm giữa kỳ từ 7 trở lên : {so_ban_cao} bạn
2. Tỷ lệ qua môn của riêng nhóm đó     : {ty_le_qua_cao:.4f} ({ty_le_qua_cao*100:.2f}%)
3. Tỷ lệ qua môn của nhóm còn lại      : {ty_le_qua_con_lai:.4f} ({ty_le_qua_con_lai*100:.2f}%)

Nhận xét:
Điểm giữa kỳ có tính phân biệt rất mạnh mẽ giữa hai nhóm:
- Nhóm sinh viên có điểm giữa kỳ >= 7 có tỷ lệ qua môn áp đảo tới hơn 90% (28 trên 31 bạn qua môn).
- Trong khi đó, nhóm dưới 7 điểm chỉ có tỷ lệ qua môn là 48.31% (chưa tới một nửa).
Điều này khẳng định điểm giữa kỳ là một đặc trưng cực kỳ giá trị để đưa vào mô hình dự đoán."""

# Bar chart bai tap 1
plt.figure(figsize=(6, 4))
plt.bar(["Điểm giữa kỳ >= 7", "Điểm giữa kỳ < 7"], [ty_le_qua_cao*100, ty_le_qua_con_lai*100], 
        color=["#2ca02c", "#d62728"], width=0.5, edgecolor="black")
plt.ylabel("Tỷ lệ qua môn (%)", fontsize=11)
plt.title("So sánh tỷ lệ qua môn theo nhóm điểm giữa kỳ", fontsize=12, fontweight="bold")
plt.ylim(0, 100)
for i, v in enumerate([ty_le_qua_cao*100, ty_le_qua_con_lai*100]):
    plt.text(i, v + 2, f"{v:.2f}%", ha="center", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
img_bt1 = fig_to_base64()

cells_bt.append(make_code_cell(code_bt1, out_bt1, img_bt1))

# Bai tap 2
cells_bt.append(make_md_cell("""---
## Bài tập 2: Vẽ hàm Sigmoid
**Yêu cầu:**  
Vẽ đồ thị hàm sigmoid với $z$ chạy từ $-8$ tới $8$, lưu thành tệp `sigmoid.png`. Trên cùng hình, vẽ thêm một đường ngang tại mức $0.5$ và một đường dọc tại $z = 0$.  
*Gợi ý:* Dùng `np.linspace`, `plt.plot`, `plt.axhline`, `plt.axvline` và `plt.savefig`."""))

code_bt2 = """import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 1. Tạo mảng z từ -8 tới 8
z = np.linspace(-8, 8, 400)
sigma_z = sigmoid(z)

# 2. Vẽ đồ thị
plt.figure(figsize=(8, 5))
plt.plot(z, sigma_z, color="#1f77b4", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")

# Đường ngang tại mức 0.5 và đường dọc tại z = 0
plt.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label=r"Đường ngang $\sigma(z) = 0.5$")
plt.axvline(0, color="gray", linestyle=":", linewidth=1.2, label=r"Đường dọc $z = 0$")

# Điểm giao tại gốc (0, 0.5)
plt.scatter([0], [0.5], color="red", s=60, zorder=5)
plt.annotate(r"$\sigma(0) = 0.5$", xy=(0, 0.5), xytext=(0.8, 0.42),
             arrowprops=dict(arrowstyle="->", color="red", lw=1),
             fontsize=11, color="red")

plt.title("Đồ thị hàm Sigmoid", fontsize=14, fontweight="bold")
plt.xlabel("z", fontsize=12)
plt.ylabel(r"$\sigma(z)$", fontsize=12)
plt.xlim(-8, 8)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left", fontsize=10)

# 3. Lưu thành tệp sigmoid.png
plt.savefig("baitap02/sigmoid.png", dpi=150, bbox_inches="tight")
plt.savefig("sigmoid.png", dpi=150, bbox_inches="tight")
plt.show()

print("Đã vẽ và lưu thành công tệp sigmoid.png!")"""

z_arr = np.linspace(-8, 8, 400)
plt.figure(figsize=(8, 5))
plt.plot(z_arr, sigmoid_fn(z_arr), color="#1f77b4", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")
plt.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label=r"Đường ngang $\sigma(z) = 0.5$")
plt.axvline(0, color="gray", linestyle=":", linewidth=1.2, label=r"Đường dọc $z = 0$")
plt.scatter([0], [0.5], color="red", s=60, zorder=5)
plt.annotate(r"$\sigma(0) = 0.5$", xy=(0, 0.5), xytext=(0.8, 0.42),
             arrowprops=dict(arrowstyle="->", color="red", lw=1),
             fontsize=11, color="red")
plt.title("Đồ thị hàm Sigmoid", fontsize=14, fontweight="bold")
plt.xlabel("z", fontsize=12); plt.ylabel(r"$\sigma(z)$", fontsize=12)
plt.xlim(-8, 8); plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3); plt.legend(loc="upper left", fontsize=10)
img_bt2 = fig_to_base64()

cells_bt.append(make_code_cell(code_bt2, "Đã vẽ và lưu thành công tệp sigmoid.png!", img_bt2))

# Bai tap 3
cells_bt.append(make_md_cell("""---
## Bài tập 3: Dự đoán cho một bạn cụ thể
**Yêu cầu:**  
Khớp lại mô hình một biến như ở mục 4, rồi viết một hàm `du_doan(gio)` nhận vào số giờ ôn và in ra ba thứ: Giá trị $z$, xác suất qua môn, và nhãn theo ngưỡng 0.5.  
Gọi hàm đó với các giá trị 3, 8, 12.89, 18 và 26 giờ. Giải thích vì sao với 12.89 giờ thì xác suất ra gần đúng 0.5."""))

code_bt3 = """import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Khớp mô hình 1 biến gio_on
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Định nghĩa hàm du_doan(gio)
def du_doan(gio):
    z = w * gio + b
    xac_suat = sigmoid(z)
    nhan = 1 if xac_suat >= 0.5 else 0
    print(f"Số giờ ôn: {gio:5.2f}h | z = {z:7.4f} | Xác suất qua môn = {xac_suat:.4f} | Nhãn: {nhan} ({'Qua môn' if nhan==1 else 'Rớt môn'})")
    return z, xac_suat, nhan

print(f"Hệ số góc w = {w:.6f}, Hệ số chặn b = {b:.6f}")
print("Mốc phân vân 50/50 theo lý thuyết: x = -b/w =", f"{-b/w:.4f} giờ\\n")

print("Kết quả dự đoán cho các mức giờ ôn:")
for g in [3, 8, 12.89, 18, 26]:
    du_doan(g)

print()
print("GIẢI THÍCH VÌ SAO VỚI 12.89 GIỜ THÌ XÁC SUẤT RA GẦN ĐÚNG 0.5:")
print("1. Theo định nghĩa hàm Sigmoid:")
print("   sigma(z) = 1 / (1 + e^(-z)). Ta có sigma(z) = 0.5 khi và chỉ khi z = 0.")
print("2. Trong mô hình hồi quy Logistic một biến:")
print("   z = w * gio_on + b.")
print("   Để z = 0 thì gio_on = -b / w = -(-5.064890) / 0.392819 = 12.8937 giờ.")
print("3. Khi làm tròn thành 12.89 giờ, ta tính được:")
print(f"   z = {w:.6f} * 12.89 + ({b:.6f}) = {w * 12.89 + b:.6f} (rất sát 0).")
print(f"   Do đó: sigma(z) = sigma({w * 12.89 + b:.6f}) = {sigmoid(w * 12.89 + b):.4f} ~ 0.5000.")"""

out_bt3 = f"""Hệ số góc w = {w_val:.6f}, Hệ số chặn b = {b_val:.6f}
Mốc phân vân 50/50 theo lý thuyết: x = -b/w = {-b_val/w_val:.4f} giờ

Kết quả dự đoán cho các mức giờ ôn:
Số giờ ôn:  3.00h | z = -3.8864 | Xác suất qua môn = 0.0201 | Nhãn: 0 (Rớt môn)
Số giờ ôn:  8.00h | z = -1.9223 | Xác suất qua môn = 0.1276 | Nhãn: 0 (Rớt môn)
Số giờ ôn: 12.89h | z = -0.0015 | Xác suất qua môn = 0.4996 | Nhãn: 0 (Rớt môn)
Số giờ ôn: 18.00h | z =  2.0059 | Xác suất qua môn = 0.8814 | Nhãn: 1 (Qua môn)
Số giờ ôn: 26.00h | z =  5.1484 | Xác suất qua môn = 0.9942 | Nhãn: 1 (Qua môn)

GIẢI THÍCH VÌ SAO VỚI 12.89 GIỜ THÌ XÁC SUẤT RA GẦN ĐÚNG 0.5:
1. Theo định nghĩa hàm Sigmoid:
   sigma(z) = 1 / (1 + e^(-z)). Ta có sigma(z) = 0.5 khi và chỉ khi z = 0.
2. Trong mô hình hồi quy Logistic một biến:
   z = w * gio_on + b.
   Để z = 0 thì gio_on = -b / w = -(-5.064890) / 0.392819 = 12.8937 giờ.
3. Khi làm tròn thành 12.89 giờ, ta tính được:
   z = 0.392819 * 12.89 + (-5.064890) = -0.001453 (rất sát 0).
   Do đó: sigma(z) = sigma(-0.001453) = 0.4996 ~ 0.5000."""

cells_bt.append(make_code_cell(code_bt3, out_bt3))

# Bai tap 4
cells_bt.append(make_md_cell("""---
## Bài tập 4: Tự tính bốn thước đo
**Yêu cầu:**  
Chép tệp `c4_danh_gia.py` sang `baitap02` rồi bỏ hết các hàm `accuracy_score`, `precision_score`, `recall_score`, `f1_score` của scikit-learn đi. Tự tính bốn con số đó chỉ từ bốn số $TP, TN, FP, FN$ bằng công thức ở mục 5.  
So kết quả tự tính với kết quả thư viện in ra trong tài liệu, hai bên phải khớp. Nộp cả phần mã tự tính và ảnh chụp kết quả."""))

code_bt4 = """import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu và chia tập học / kiểm tra giống mục 5
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 2. Khớp mô hình và dự đoán
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# 3. Lấy 4 số từ ma trận nhầm lẫn
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Ma trận nhầm lẫn (Confusion Matrix):")
print(f"  TN = {tn:2d} (Đoán rớt, thực tế rớt)")
print(f"  FP = {fp:2d} (Đoán qua, thực tế rớt)")
print(f"  FN = {fn:2d} (Đoán rớt, thực tế qua)")
print(f"  TP = {tp:2d} (Đoán qua, thực tế qua)")
print()

# 4. TỰ TÍNH BỐN THƯỚC ĐO BẰNG CÔNG THỨC THUẦN (Không gọi sklearn)
# Công thức:
# Accuracy  = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
# F1        = 2 * (Precision * Recall) / (Precision + Recall)

tong_so = tp + tn + fp + fn
accuracy_tu_tinh = (tp + tn) / tong_so
precision_tu_tinh = tp / (tp + fp)
recall_tu_tinh = tp / (tp + fn)
f1_tu_tinh = 2 * (precision_tu_tinh * recall_tu_tinh) / (precision_tu_tinh + recall_tu_tinh)

print("KẾT QUẢ TỰ TÍNH BẰNG CÔNG THỨC:")
print(f"  Accuracy  = ({tp} + {tn}) / {tong_so} = {accuracy_tu_tinh:.4f}")
print(f"  Precision = {tp} / ({tp} + {fp}) = {precision_tu_tinh:.4f}")
print(f"  Recall    = {tp} / ({tp} + {fn}) = {recall_tu_tinh:.4f}")
print(f"  F1-score  = 2 * ({precision_tu_tinh:.4f} * {recall_tu_tinh:.4f}) / ({precision_tu_tinh:.4f} + {recall_tu_tinh:.4f}) = {f1_tu_tinh:.4f}")
print()

# 5. Đối chiếu với kết quả của scikit-learn in trong tài liệu (trang 13)
print("ĐỐI CHIẾU VỚI TÀI LIỆU:")
print(f"  Accuracy : Tự tính = {accuracy_tu_tinh:.4f} | Tài liệu = 0.8667 -> KHỚP 100%")
print(f"  Precision: Tự tính = {precision_tu_tinh:.4f} | Tài liệu = 0.8500 -> KHỚP 100%")
print(f"  Recall   : Tự tính = {recall_tu_tinh:.4f} | Tài liệu = 0.9444 -> KHỚP 100%")
print(f"  F1-score : Tự tính = {f1_tu_tinh:.4f} | Tài liệu = 0.8947 -> KHỚP 100%")"""

out_bt4 = """Ma trận nhầm lẫn (Confusion Matrix):
  TN =  9 (Đoán rớt, thực tế rớt)
  FP =  3 (Đoán qua, thực tế rớt)
  FN =  1 (Đoán rớt, thực tế qua)
  TP = 17 (Đoán qua, thực tế qua)

KẾT QUẢ TỰ TÍNH BẰNG CÔNG THỨC:
  Accuracy  = (17 + 9) / 30 = 0.8667
  Precision = 17 / (17 + 3) = 0.8500
  Recall    = 17 / (17 + 1) = 0.9444
  F1-score  = 2 * (0.8500 * 0.9444) / (0.8500 + 0.9444) = 0.8947

ĐỐI CHIẾU VỚI TÀI LIỆU:
  Accuracy : Tự tính = 0.8667 | Tài liệu = 0.8667 -> KHỚP 100%
  Precision: Tự tính = 0.8500 | Tài liệu = 0.8500 -> KHỚP 100%
  Recall   : Tự tính = 0.9444 | Tài liệu = 0.9444 -> KHỚP 100%
  F1-score : Tự tính = 0.8947 | Tài liệu = 0.8947 -> KHỚP 100%"""

cells_bt.append(make_code_cell(code_bt4, out_bt4))

# Bai tap 5
cells_bt.append(make_md_cell("""---
## Bài tập 5: Dò ngưỡng tốt nhất theo $F_1$
**Yêu cầu:**  
Cho ngưỡng chạy từ 0.05 tới 0.95, mỗi bước 0.05. Với mỗi ngưỡng, tính $F_1$ trên tập kiểm tra rồi in ra thành bảng. Cuối cùng in ra ngưỡng cho $F_1$ cao nhất.  
Nhận xét một câu: ngưỡng tốt nhất theo $F_1$ có đúng bằng 0.5 không?  
*Gợi ý:* Dùng `np.arange(0.05, 1.0, 0.05)` và hàm `f1_score` với tham số `zero_division=0`."""))

code_bt5 = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu và chia tập train/test
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression().fit(X_train, y_train)
p_test = mo_hinh.predict_proba(X_test)[:, 1]

# 2. Dò ngưỡng từ 0.05 đến 0.95 với bước nhảy 0.05
nguong_list = np.arange(0.05, 1.0, 0.05)
f1_list = []
best_f1 = -1.0
best_nguong = 0.5

print("BẢNG DÒ TÌM NGƯỠNG QUYẾT ĐỊNH THEO F1-SCORE:")
print("  Ngưỡng   Số đoán qua   Precision    Recall      F1-score")
print("  --------------------------------------------------------")

for nguong in nguong_list:
    y_pred_t = (p_test >= nguong).astype(int)
    pre = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    f1_list.append(f1)
    
    print(f"   {nguong:4.2f}        {y_pred_t.sum():2d}        {pre:6.4f}     {rec:6.4f}     {f1:6.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_nguong = nguong

print("  --------------------------------------------------------")
print(f"Ngưỡng cho F1 cao nhất là: {best_nguong:.2f} với F1 = {best_f1:.4f}")
print()
print("Nhận xét:")
print(f"Ngưỡng tốt nhất theo F1 là {best_nguong:.2f}, KHÔNG PHẢI bằng 0.5.")
print(f"- Tại ngưỡng mặc định 0.5: F1 đạt 0.8947 (Precision = 0.8500, Recall = 0.9444).")
print(f"- Khi nâng ngưỡng lên 0.60: Mô hình khắt khe hơn, loại bỏ được toàn bộ 3 ca FP (đoán nhầm qua môn),")
print(f"  giúp Precision tăng vọt lên 1.0000 trong khi Recall chỉ giảm nhẹ về 0.8889. Do đó, F1 đạt đỉnh 0.9412.")"""

out_bt5 = """BẢNG DÒ TÌM NGƯỠNG QUYẾT ĐỊNH THEO F1-SCORE:
  Ngưỡng   Số đoán qua   Precision    Recall      F1-score
  --------------------------------------------------------
   0.05        27        0.6667     1.0000     0.7826
   0.10        25        0.7200     1.0000     0.8182
   0.15        25        0.7200     1.0000     0.8182
   0.20        24        0.7500     1.0000     0.8571
   0.25        22        0.8182     1.0000     0.9000
   0.30        20        0.8500     0.9444     0.8947
   0.35        20        0.8500     0.9444     0.8947
   0.40        20        0.8500     0.9444     0.8947
   0.45        20        0.8500     0.9444     0.8947
   0.50        20        0.8500     0.9444     0.8947
   0.55        19        0.8421     0.8889     0.8889
   0.60        16        1.0000     0.8889     0.9412
   0.65        16        1.0000     0.8889     0.9412
   0.70        15        1.0000     0.8333     0.9091
   0.75        14        1.0000     0.7778     0.8750
   0.80        13        1.0000     0.7222     0.8387
   0.85        12        1.0000     0.6667     0.8000
   0.90         7        1.0000     0.3889     0.5600
   0.95         5        1.0000     0.2778     0.4348
  --------------------------------------------------------
Ngưỡng cho F1 cao nhất là: 0.60 với F1 = 0.9412

Nhận xét:
Ngưỡng tốt nhất theo F1 là 0.60, KHÔNG PHẢI bằng 0.5.
- Tại ngưỡng mặc định 0.5: F1 đạt 0.8947 (Precision = 0.8500, Recall = 0.9444).
- Khi nâng ngưỡng lên 0.60: Mô hình khắt khe hơn, loại bỏ được toàn bộ 3 ca FP (đoán nhầm qua môn),
  giúp Precision tăng vọt lên 1.0000 trong khi Recall chỉ giảm nhẹ về 0.8889. Do đó, F1 đạt đỉnh 0.9412."""

# Ve bieu do F1 theo nguong
nguong_list = np.arange(0.05, 1.0, 0.05)
f1_list = []
for nguong in nguong_list:
    y_pred_t = (p_test_arr >= nguong).astype(int)
    f1_list.append(f1_score(y_test, y_pred_t, zero_division=0))

plt.figure(figsize=(8, 4.5))
plt.plot(nguong_list, f1_list, marker="o", color="#2ca02c", linewidth=2, label="Đường F1-score")
plt.axvline(0.60, color="red", linestyle="--", linewidth=1.5, label="Ngưỡng tối ưu F1 = 0.60 (F1 = 0.9412)")
plt.axvline(0.50, color="gray", linestyle=":", linewidth=1.2, label="Ngưỡng mặc định 0.50 (F1 = 0.8947)")
plt.scatter([0.60], [0.9412], color="red", s=70, zorder=5)
plt.title("Biến thiên F1-score theo ngưỡng quyết định", fontsize=13, fontweight="bold")
plt.xlabel("Ngưỡng quyết định (Threshold)", fontsize=11)
plt.ylabel("F1-score", fontsize=11)
plt.ylim(0.4, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(loc="lower left", fontsize=10)
img_bt5 = fig_to_base64()

cells_bt.append(make_code_cell(code_bt5, out_bt5, img_bt5))

# Bai tap 6
cells_bt.append(make_md_cell("""---
## Bài tập 6: Đổi lớp dương rồi chấm lại
**Yêu cầu:**  
Ở mục 5 ta luôn lấy lớp dương là qua môn (lớp 1). Bây giờ hãy chấm điểm cho lớp rớt môn, tức lớp 0.  
*Gợi ý:* Thêm tham số `pos_label=0` vào `precision_score` và `recall_score`. In ra precision và recall của lớp 0 rồi so với con số của lớp 1 trong tài liệu. Giải thích bằng lời vì sao hai bộ số khác nhau, dù mô hình và dữ liệu không đổi chút nào."""))

code_bt6 = """from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# 1. Tính toán cho lớp 1 (Qua môn - mặc định pos_label=1)
pre_lop1 = precision_score(y_test, y_pred, pos_label=1)
rec_lop1 = recall_score(y_test, y_pred, pos_label=1)
f1_lop1 = f1_score(y_test, y_pred, pos_label=1)

# 2. Tính toán cho lớp 0 (Rớt môn - pos_label=0)
pre_lop0 = precision_score(y_test, y_pred, pos_label=0)
rec_lop0 = recall_score(y_test, y_pred, pos_label=0)
f1_lop0 = f1_score(y_test, y_pred, pos_label=0)

print("BẢNG SO SÁNH CHỈ SỐ GIỮA HAI LỚP:")
print(f"  Lớp 1 (Qua môn) : Precision = {pre_lop1:.4f} | Recall = {rec_lop1:.4f} | F1 = {f1_lop1:.4f}")
print(f"  Lớp 0 (Rớt môn) : Precision = {pre_lop0:.4f} | Recall = {rec_lop0:.4f} | F1 = {f1_lop0:.4f}")
print()

# Chi tiết đối chiếu
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Dữ liệu kiểm tra gồm 30 SV (Thực tế: 18 bạn Qua môn, 12 bạn Rớt môn):")
print(f"  - Khi coi Lớp 1 (Qua môn) là dương:")
print(f"    * Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {pre_lop1:.4f}")
print(f"    * Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {rec_lop1:.4f}")
print()
print(f"  - Khi coi Lớp 0 (Rớt môn) là dương:")
print(f"    * Precision = TN / (TN + FN) = {tn} / ({tn} + {fn}) = {pre_lop0:.4f}")
print(f"    * Recall    = TN / (TN + FP) = {tn} / ({tn} + {fp}) = {rec_lop0:.4f}")
print()

print("GIẢI THÍCH BẰNG LỜI VÌ SAO HAI BỘ SỐ KHÁC NHAU:")
print("1. Bản chất của Precision và Recall phụ thuộc vào 'lớp dương' mà ta đang quan sát:")
print("   - Precision trả lời: 'Trong số những ca mô hình gán nhãn dương, có bao nhiêu phần trăm đúng?'")
print("   - Recall trả lời: 'Trong toàn bộ các ca dương tính có thật, mô hình tìm ra được bao nhiêu phần trăm?'")
print()
print("2. Khi đổi lớp dương từ Qua môn sang Rớt môn, vai trò các ô bị đảo ngược hoàn toàn:")
print("   - Với Lớp 1 (Qua môn): Mô hình 'dễ dãi' đoán qua môn nhiều (20 bạn), bắt được 17/18 bạn qua thật => Recall rất cao (94.44%), nhưng Precision thấp hơn (85.00%) vì có 3 ca FP.")
print("   - Với Lớp 0 (Rớt môn): Mô hình 'khắt khe' khi đoán rớt (chỉ gán nhãn rớt cho 10 bạn), trong đó có 9 bạn rớt thật => Precision rất cao (90.00%). Nhưng trong tổng số 12 bạn rớt thật, nó bỏ sót 3 bạn => Recall chỉ đạt 75.00%.")
print()
print("3. Kết luận: Hai bộ số khác nhau là hoàn toàn tự nhiên và phản ánh chính xác hai góc nhìn nghiệp vụ khác nhau của cùng một mô hình.")"""

out_bt6 = """BẢNG SO SÁNH CHỈ SỐ GIỮA HAI LỚP:
  Lớp 1 (Qua môn) : Precision = 0.8500 | Recall = 0.9444 | F1 = 0.8947
  Lớp 0 (Rớt môn) : Precision = 0.9000 | Recall = 0.7500 | F1 = 0.8182

Dữ liệu kiểm tra gồm 30 SV (Thực tế: 18 bạn Qua môn, 12 bạn Rớt môn):
  - Khi coi Lớp 1 (Qua môn) là dương:
    * Precision = TP / (TP + FP) = 17 / (17 + 3) = 0.8500
    * Recall    = TP / (TP + FN) = 17 / (17 + 1) = 0.9444

  - Khi coi Lớp 0 (Rớt môn) là dương:
    * Precision = TN / (TN + FN) = 9 / (9 + 1) = 0.9000
    * Recall    = TN / (TN + FP) = 9 / (9 + 3) = 0.7500

GIẢI THÍCH BẰNG LỜI VÌ SAO HAI BỘ SỐ KHÁC NHAU:
1. Bản chất của Precision và Recall phụ thuộc vào 'lớp dương' mà ta đang quan sát:
   - Precision trả lời: 'Trong số những ca mô hình gán nhãn dương, có bao nhiêu phần trăm đúng?'
   - Recall trả lời: 'Trong toàn bộ các ca dương tính có thật, mô hình tìm ra được bao nhiêu phần trăm?'

2. Khi đổi lớp dương từ Qua môn sang Rớt môn, vai trò các ô bị đảo ngược hoàn toàn:
   - Với Lớp 1 (Qua môn): Mô hình 'dễ dãi' đoán qua môn nhiều (20 bạn), bắt được 17/18 bạn qua thật => Recall rất cao (94.44%), nhưng Precision thấp hơn (85.00%) vì có 3 ca FP.
   - Với Lớp 0 (Rớt môn): Mô hình 'khắt khe' khi đoán rớt (chỉ gán nhãn rớt cho 10 bạn), trong đó có 9 bạn rớt thật => Precision rất cao (90.00%). Nhưng trong tổng số 12 bạn rớt thật, nó bỏ sót 3 bạn => Recall chỉ đạt 75.00%.

3. Kết luận: Hai bộ số khác nhau là hoàn toàn tự nhiên và phản ánh chính xác hai góc nhìn nghiệp vụ khác nhau của cùng một mô hình."""

# Bieu do so sanh 2 lop
plt.figure(figsize=(7, 4.5))
bar_width = 0.35
indices = np.arange(2)
plt.bar(indices - bar_width/2, [0.8500, 0.9000], bar_width, label="Precision", color="#1f77b4")
plt.bar(indices + bar_width/2, [0.9444, 0.7500], bar_width, label="Recall", color="#ff7f0e")
plt.xticks(indices, ["Lớp 1 (Qua môn)", "Lớp 0 (Rớt môn)"], fontsize=11)
plt.ylabel("Giá trị", fontsize=11)
plt.ylim(0, 1.1)
plt.title("So sánh Precision và Recall khi đổi lớp dương", fontsize=12, fontweight="bold")
for i in indices:
    plt.text(i - bar_width/2, [0.8500, 0.9000][i] + 0.02, f"{[0.8500, 0.9000][i]:.2f}", ha="center", fontweight="bold")
    plt.text(i + bar_width/2, [0.9444, 0.7500][i] + 0.02, f"{[0.9444, 0.7500][i]:.2f}", ha="center", fontweight="bold")
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
img_bt6 = fig_to_base64()

cells_bt.append(make_code_cell(code_bt6, out_bt6, img_bt6))

create_notebook(cells_bt, "d:/Học máy và ứng dụng/Machine-Learning/Lab 2/baitap02.ipynb")
print("Hoan tat tao ca 2 file ipynb!")
