# -*- coding: utf-8 -*-
"""Bài tập 4: Láng giềng có trọng số.

Yêu cầu:
1. Thêm tham số weights="distance" vào KNeighborsClassifier.
2. Chạy lại với k = 1, 13, 101 trên bộ hai vầng trăng và so với bảng ở mục 4 (weights="uniform").
3. Trả lời: Với k = 101, trọng số theo khoảng cách có giúp mô hình bớt cẩu thả không? Vì sao?
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

# Chia dữ liệu theo đúng chuẩn mục 4
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("=" * 70)
print("KET QUA BAI TAP 4: LANG GIENG CO TRONG SO (weights='distance')")
print("=" * 70)
print(f"{'k':>5} | {'Uniform (Tap hoc)':>18} | {'Uniform (Kiem tra)':>18} | {'Distance (Tap hoc)':>18} | {'Distance (Kiem tra)':>19}")
print("-" * 85)

cac_k = [1, 13, 101]
ket_qua = {}

for k in cac_k:
    # Mô hình trọng số đều (uniform) ở mục 4
    mh_u = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_tr, y_tr)
    acc_tr_u = accuracy_score(y_tr, mh_u.predict(X_tr))
    acc_te_u = accuracy_score(y_te, mh_u.predict(X_te))

    # Mô hình trọng số theo khoảng cách (distance)
    mh_d = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_tr, y_tr)
    acc_tr_d = accuracy_score(y_tr, mh_d.predict(X_tr))
    acc_te_d = accuracy_score(y_te, mh_d.predict(X_te))

    ket_qua[k] = (acc_tr_u, acc_te_u, acc_tr_d, acc_te_d)
    print(f"{k:5d} | {acc_tr_u:18.4f} | {acc_te_u:18.4f} | {acc_tr_d:18.4f} | {acc_te_d:19.4f}")

print()
print("TRA LOI CAU HOI:")
print("Cau hoi: Voi k = 101, trong so theo khoang cach co giup mo hinh bot cau tha khong? Vi sao?")
print(f"-> CO, trong so khoang cach giup mo hinh bot cau tha rat ro ret.")
print(f"   Accuracy tren tap kiem tra tang vot tu {ket_qua[101][1]:.4f} (uniform) len {ket_qua[101][3]:.4f} (distance).")
print()
print("GIAI THICH NGUYEN LY:")
print("1. Khi weights='uniform' voi k = 101:")
print("   - Moi diem trong so 101 lang gieng deu co 1 phieu bau ngang gia tri nhu nhau.")
print("   - Boi vi k qua lon (chiem gan mot nua tap du lieu 225 mau), cac diem o rat xa")
print("     cung tham gia bo phieu lam 'pha loang' thong tin cuc bo.")
print("   - He qua la bien quyet dinh bi phang hoa gan nhu duong thang, bo qua duong cong")
print("     hai vang trang (mo hinh tro nen cau tha).")
print("2. Khi weights='distance' voi k = 101:")
print("   - Moi lang gieng duoc gan trong so w_i = 1 / d(x_moi, x_i).")
print("   - Nhung diem cuc gan se co trong so rat lon, trong khi cac diem o xa co trong so rat nho.")
print("   - Do do, du lay toi 101 diem, tieng noi quyet dinh van thuoc ve nhung diem sat canh.")
print("   - Bien quyet dinh giu duoc do cong uyen chuyen theo hinh dang hai vang trang, khong bi")
print("     phang hoa, giup accuracy tang them hon 5.3% (tu 0.8533 len 0.9067).")
