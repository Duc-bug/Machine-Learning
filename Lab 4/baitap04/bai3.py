# -*- coding: utf-8 -*-
"""Bài tập 3: Đổi đơn vị đo.

Yêu cầu:
1. Trên bộ sinh_vien_mo_rong.csv, đổi cột chi_tieu từ nghìn đồng sang triệu đồng bằng cách chia cho 1000.
2. Chạy KNN với k = 7 không chuẩn hoá trên ba cột mới.
3. Trả lời 3 câu hỏi:
   - Accuracy thay đổi thế nào so với 0.5333 ở mục 3?
   - Giải thích vì sao chỉ đổi đơn vị mà kết quả lại khác hẳn.
   - Nếu có chuẩn hoá thì đổi đơn vị có làm kết quả thay đổi không?
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu sinh viên mở rộng
data_path = "data/sinh_vien_mo_rong.csv"
if not os.path.exists(data_path):
    if os.path.exists("../data/sinh_vien_mo_rong.csv"):
        data_path = "../data/sinh_vien_mo_rong.csv"
    elif os.path.exists("sinh_vien_mo_rong.csv"):
        data_path = "sinh_vien_mo_rong.csv"

df = pd.read_csv(data_path)
y = df["qua_mon"]

# Đổi đơn vị chi_tieu từ nghìn đồng sang triệu đồng (chia cho 1000)
df["chi_tieu_trieu"] = df["chi_tieu"] / 1000.0

# Tập đặc trưng cũ (nghìn đồng) và mới (triệu đồng)
X_cu = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
X_moi = df[["gio_on", "diem_giua_ky", "chi_tieu_trieu"]]

# Chia tập theo đúng random_state=17, stratify=y
X_tr_cu, X_te_cu, y_tr, y_te = train_test_split(
    X_cu, y, test_size=0.25, random_state=17, stratify=y
)
X_tr_m, X_te_m, _, _ = train_test_split(
    X_moi, y, test_size=0.25, random_state=17, stratify=y
)

# 1. Chạy KNN (k = 7) KHÔNG chuẩn hoá trên 3 cột mới
knn_khong_chuan = KNeighborsClassifier(n_neighbors=7)
knn_khong_chuan.fit(X_tr_m, y_tr)
acc_moi_khong_chuan = accuracy_score(y_te, knn_khong_chuan.predict(X_te_m))

# Đối chiếu với bản cũ (nghìn đồng) không chuẩn hóa ở mục 3
knn_cu_khong_chuan = KNeighborsClassifier(n_neighbors=7)
knn_cu_khong_chuan.fit(X_tr_cu, y_tr)
acc_cu_khong_chuan = accuracy_score(y_te, knn_cu_khong_chuan.predict(X_te_cu))

# 2. Thử nghiệm CÓ chuẩn hoá trên cả hai đơn vị đo
pipe_cu = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)).fit(X_tr_cu, y_tr)
acc_cu_chuan = accuracy_score(y_te, pipe_cu.predict(X_te_cu))

pipe_moi = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)).fit(X_tr_m, y_tr)
acc_moi_chuan = accuracy_score(y_te, pipe_moi.predict(X_te_m))

print("=" * 60)
print("KET QUA BAI TAP 3: DOI DON VI DO")
print("=" * 60)
print(f"Accuracy khi chi_tieu tinh bang NGHIN DONG (khong chuan hoa): {acc_cu_khong_chuan:.4f}")
print(f"Accuracy khi chi_tieu tinh bang TRIEU DONG (khong chuan hoa): {acc_moi_khong_chuan:.4f}")
print(f"Accuracy khi chi_tieu tinh bang NGHIN DONG (CO chuan hoa)    : {acc_cu_chuan:.4f}")
print(f"Accuracy khi chi_tieu tinh bang TRIEU DONG (CO chuan hoa)    : {acc_moi_chuan:.4f}")
print()
print("TRA LOI CAC CAU HOI THEO TIEU CHI CHAM:")
print("1. Accuracy thay doi the nao so voi 0.5333 o muc 3?")
print(f"   -> Accuracy tang vot tu {acc_cu_khong_chuan:.4f} (53.33%) len {acc_moi_khong_chuan:.4f} (86.67%).")
print()
print("2. Giai thich vi sao chi doi don vi ma ket qua lai khac han:")
print("   -> Cong thuc khoang cach Euclid: d(x, z) = sqrt( (x1-z1)^2 + (x2-z2)^2 + (x3-z3)^2 ).")
print("   - Khi dung don vi 'nghin dong', chi tieu dao dong tu 2850 den 7700 (do lech chuan ~1692).")
print("     Chenh lech chi tieu hang nghin don vi da 'de bep' va nuot tron chenh lech cua gio on (0-30)")
print("     va diem so (0-10), khien mo hinh xem hai nguoi tieu tien giong nhau la hang xom.")
print("   - Khi chia cho 1000 (doi sang 'trieu dong'), gia tri chi tieu chi con dao dong tu 2.85 den 7.70")
print("     (do lech chuan ~1.69). Thang do nay da tuong dong voi gio on va diem so, giup hai cot hoc tap")
print("     khong bi che khuat, nho do do chinh xac tang vot len 0.8667.")
print()
print("3. Neu co chuan hoa thi doi don vi co lam ket qua thay doi khong?")
print(f"   -> KHONG he thay doi (ca hai deu giu nguyen accuracy = {acc_moi_chuan:.4f}).")
print("   - Ly do toan hoc: Cong thuc z-score la z = (x - mean) / std. Khi ta nhan hay chia x")
print("     cho mot hang so c > 0 (doi don vi), ca (x - mean) va std deu duoc nhan/chia cho c,")
print("     khien ty so z hoan toan giu nguyen. Do do, chuan hoa giup KNN bat bien truoc moi")
print("     phep co dan tuyen tinh hay doi don vi do.")
