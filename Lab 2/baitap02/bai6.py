# -*- coding: utf-8 -*-
"""
Bài tập 6: Đổi lớp dương rồi chấm lại
Ở mục 5 ta luôn lấy lớp dương là qua môn. Bây giờ hãy chấm điểm cho lớp rớt môn, tức lớp 0.
Gợi ý: thêm tham số pos_label=0 vào precision_score và recall_score.
In ra precision và recall của lớp 0 rồi so với con số của lớp 1 trong tài liệu.
Giải thích bằng lời vì sao hai bộ số khác nhau, dù mô hình và dữ liệu không đổi chút nào.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu và chia tập train/test giống mục 5
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

y_pred = mo_hinh.predict(X_test)

# 2. Tính cho lớp 1 (Qua môn - mặc định pos_label=1)
pre_lop1 = precision_score(y_test, y_pred, pos_label=1)
rec_lop1 = recall_score(y_test, y_pred, pos_label=1)
f1_lop1 = f1_score(y_test, y_pred, pos_label=1)

# 3. Tính cho lớp 0 (Rớt môn - đặt pos_label=0)
pre_lop0 = precision_score(y_test, y_pred, pos_label=0)
rec_lop0 = recall_score(y_test, y_pred, pos_label=0)
f1_lop0 = f1_score(y_test, y_pred, pos_label=0)

print("So sanh chi so giua hai lop:")
print(f"  Lop 1 (Qua mon) : Precision = {pre_lop1:.4f} | Recall = {rec_lop1:.4f} | F1 = {f1_lop1:.4f}")
print(f"  Lop 0 (Rot mon) : Precision = {pre_lop0:.4f} | Recall = {rec_lop0:.4f} | F1 = {f1_lop0:.4f}")
print()

# In ma trận nhầm lẫn để đối chiếu trực quan
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Chi tiet tren 30 ban tap kiem tra (TN={tn}, FP={fp}, FN={fn}, TP={tp}):")
print(f"  - Lop 0 (Rot mon): Co tat ca {tn + fp} ban that su rot mon. Mo hinh doan rot {tn + fn} ban, trong do dung {tn} ban.")
print(f"    * Precision lop 0 = TN / (TN + FN) = {tn} / ({tn} + {fn}) = {tn / (tn + fn):.4f}")
print(f"    * Recall lop 0    = TN / (TN + FP) = {tn} / ({tn} + {fp}) = {tn / (tn + fp):.4f}")
print()

print("GIAI THICH BANG LOI VI SAO HAI BO SO KHAC NHAU:")
print("1. Ban chat cua Precision va Recall phu thuoc vao viec ta xem lop nao la 'duong tinh' (positive):")
print("   - Khi chon lop 1 (Qua mon) lam lop duong:")
print("     * Precision do: Trong so nhung ban duoc mo hinh doan la QUA, co bao nhieu ban qua that (17/20 = 85%).")
print("     * Recall do: Trong so tat ca cac ban QUA THAT, mo hinh tim ra duoc bao nhieu ban (17/18 = 94.44%).")
print()
print("   - Khi chon lop 0 (Rot mon) lam lop duong:")
print("     * Vai tro cac o bi doi cho: TN (doan rot dung) tro thanh True Positive cua lop 0.")
print("     * Precision do: Trong so nhung ban duoc mo hinh doan la ROT, co bao nhieu ban rot that (9/10 = 90%).")
print("     * Recall do: Trong so tat ca cac ban ROT THAT (12 ban), mo hinh chi bat duoc 9 ban, bo sot 3 ban (9/12 = 75%).")
print()
print("2. Ket luan:")
print("   Mo hinh nay co xu huong 'de tinh' cho qua mon (nghieng ve doan qua mon):")
print("   - No bat duoc rat tot cac ban qua mon (Recall lop 1 rat cao: 94.44%),")
print("   - Nhung doi lai bo sot 3 ban rot mon (Recall lop 0 chi dat 75%).")
print("   - Khi mo hinh da doan ai rot thi kha nang cao ban do rot that (Precision lop 0 dat toi 90%).")
print("   Do do, cung mot mo hinh va du lieu, hai bo so phai khac nhau vi chung danh gia hai cau hoi nghiep vu hoan toan khac nhau.")
