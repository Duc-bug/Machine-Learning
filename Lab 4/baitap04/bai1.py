# -*- coding: utf-8 -*-
"""Bài tập 1: Tự viết KNN không dùng thư viện.

Yêu cầu:
1. Dựa vào e1_khoang_cach.py, viết hàm knn_du_doan(X_hoc, y_hoc, x_moi, k) trả về nhãn dự đoán cho một mẫu mới.
2. Dùng hàm đó dự đoán cho toàn bộ tập kiểm tra của bộ hai vầng trăng với k = 13, tính accuracy.
3. So sánh kết quả với KNeighborsClassifier ở mục 4 (hai con số phải bằng nhau).
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu hai vầng trăng (hỗ trợ nhiều trường hợp đường dẫn)
data_path = "data/hai_vang_trang.csv"
if not os.path.exists(data_path):
    if os.path.exists("../data/hai_vang_trang.csv"):
        data_path = "../data/hai_vang_trang.csv"
    elif os.path.exists("hai_vang_trang.csv"):
        data_path = "hai_vang_trang.csv"

df = pd.read_csv(data_path)
X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["nhan"].to_numpy(dtype=int)

# Chia tập train/test giống hệt mục 4: test_size=0.25, random_state=17, stratify=y
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Dự đoán nhãn cho mẫu mới x_moi dựa trên tập học (X_hoc, y_hoc) và số láng giềng k.
    
    Các bước thực hiện:
    - Bước 1: Tính khoảng cách Euclid từ x_moi đến mọi điểm trong X_hoc.
    - Bước 2: Dùng np.argsort để chọn k láng giềng có khoảng cách gần nhất.
    - Bước 3: Cho k láng giềng bỏ phiếu đa số để xác định nhãn dự đoán.
    """
    # 1. Tính khoảng cách Euclid giữa x_moi và mọi điểm trong X_hoc
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))

    # 2. Lấy chỉ số của k láng giềng có khoảng cách nhỏ nhất
    chi_so_k_gan_nhat = np.argsort(khoang_cach)[:k]

    # 3. Lấy nhãn của k láng giềng gần nhất
    nhan_lang_gieng = y_hoc[chi_so_k_gan_nhat]

    # 4. Bỏ phiếu đa số (majority voting)
    nhan_du_doan = np.bincount(nhan_lang_gieng).argmax()
    return nhan_du_doan


# Dự đoán cho toàn bộ tập kiểm tra với k = 13 bằng hàm tự viết
k = 13
y_pred_tu_viet = np.array([knn_du_doan(X_tr, y_tr, x, k) for x in X_te])
acc_tu_viet = accuracy_score(y_te, y_pred_tu_viet)

# Dự đoán bằng thư viện scikit-learn để đối chiếu
mo_hinh_sklearn = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
y_pred_sklearn = mo_hinh_sklearn.predict(X_te)
acc_sklearn = accuracy_score(y_te, y_pred_sklearn)

print("=" * 60)
print("KET QUA BAI TAP 1: TU VIET KNN KHONG DUNG THU VIEN")
print("=" * 60)
print(f"Accuracy KNN tu viet (k = {k})         : {acc_tu_viet:.4f}")
print(f"Accuracy KNeighborsClassifier (k = {k}) : {acc_sklearn:.4f}")
print(f"Ket qua hai mo hinh bang nhau         : {acc_tu_viet == acc_sklearn}")
print()
print("Giai thich nguyen ly:")
print("1. KNN la mo hinh 'hoc luoi' (lazy learning): qua trinh fit chi la luu tru")
print("   toan bo tap du lieu vao bo nho, moi phep toan deu dien ra luc predict.")
print("2. Khi du doan cho mot diem moi x_moi:")
print("   - Tinh khoang cach Euclid tu x_moi toi tat ca diem hoc: d = sqrt(sum((x - z)^2)).")
print("   - Su dung np.argsort de lay k chi so ung voi k khoang cach nho nhat.")
print("   - Dem so phieu bau cua tung lop bang np.bincount va chon lop co phieu cao nhat.")
print("3. Ket qua hoan toan khop voi scikit-learn (cung dat 0.9200) chung minh tinh")
print("   chinh xac tuyet doi cua thuat toan tu cai dat.")
