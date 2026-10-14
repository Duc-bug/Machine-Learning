# -*- coding: utf-8 -*-
"""Bài tập 2: Khoảng cách Manhattan.

Yêu cầu:
1. Chạy lại e3_chon_k.py nhưng thêm tham số metric="manhattan" vào KNeighborsClassifier.
2. Ghi lại k tốt nhất và accuracy trên tập kiểm tra, so với khoảng cách Euclid.
3. Nhận xét một câu: đổi cách đo khoảng cách có làm kết quả khác đi nhiều không?
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Đọc dữ liệu hai vầng trăng
data_path = "data/hai_vang_trang.csv"
if not os.path.exists(data_path):
    if os.path.exists("../data/hai_vang_trang.csv"):
        data_path = "../data/hai_vang_trang.csv"
    elif os.path.exists("hai_vang_trang.csv"):
        data_path = "hai_vang_trang.csv"

df = pd.read_csv(data_path)
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia tập theo mục 4
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=" * 60)
print("KET QUA BAI TAP 2: KHOANG CACH MANHATTAN")
print("=" * 60)

# 1. Chạy kiểm định chéo 5 phần với metric="manhattan"
print("Kiem dinh cheo 5 phan tren tap hoc (Manhattan):")
diem_manhattan = {}
for k in range(1, 52, 2):
    diem_manhattan[k] = cross_val_score(
        KNeighborsClassifier(n_neighbors=k, metric="manhattan"),
        X_tr, y_tr, cv=5
    ).mean()

k_tot_manhattan = max(diem_manhattan, key=diem_manhattan.get)
for k in [1, 5, 9, 13, 17, 25, 51]:
    dau = " <- tot nhat" if k == k_tot_manhattan else ""
    print(f"  k = {k:2d}  diem trung binh: {diem_manhattan[k]:.4f}{dau}")

# Đánh giá trên tập kiểm tra với k tốt nhất (Manhattan)
mh_manhattan = KNeighborsClassifier(n_neighbors=k_tot_manhattan, metric="manhattan").fit(X_tr, y_tr)
acc_test_manhattan = accuracy_score(y_te, mh_manhattan.predict(X_te))

# 2. Đối chiếu với Euclid (k = 13 theo mục 4)
mh_euclid = KNeighborsClassifier(n_neighbors=13, metric="euclidean").fit(X_tr, y_tr)
acc_test_euclid = accuracy_score(y_te, mh_euclid.predict(X_te))

print()
print("SO SANH GIUA MANHATTAN VA EUCLID:")
print(f"- Manhattan: k tot nhat = {k_tot_manhattan:2d}, CV Score = {diem_manhattan[k_tot_manhattan]:.4f}, Test Accuracy = {acc_test_manhattan:.4f}")
print(f"- Euclid   : k tot nhat = 13, CV Score = 0.8667, Test Accuracy = {acc_test_euclid:.4f}")
print()
print("NHAN XET:")
print("Doi cach do khoang cach tu Euclid sang Manhattan khong lam ket qua khac di nhieu")
print(f"(accuracy tap kiem tra van dat muc cao rat tuong duong nhau: {acc_test_manhattan:.4f} so voi {acc_test_euclid:.4f}),")
print("vi cau truc hinh hoc cua bo du lieu hai vang trang 2 chieu van bao toan thu tu cac lang gieng gan nhat.")
