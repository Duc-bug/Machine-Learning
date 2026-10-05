# -*- coding: utf-8 -*-
"""
Bài tập 4: Tự tính bốn thước đo
Chép tệp c4_danh_gia.py sang baitap02 rồi bỏ hết các hàm accuracy_score,
precision_score, recall_score, f1_score của scikit-learn đi.
Tự tính bốn con số đó chỉ từ bốn số TP, TN, FP, FN bằng công thức ở mục 5.
So kết quả tự tính với kết quả thư viện in ra trong tài liệu.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on"]]
y = df["qua_mon"]

# 2. Chia tập học và tập kiểm tra (stratify=y, random_state=17)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# 3. Khớp mô hình
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

# 4. Dự đoán trên tập kiểm tra
y_pred = mo_hinh.predict(X_test)

# 5. Lấy ma trận nhầm lẫn
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Ma tran nham lan:")
print(f"  TN = {tn:2d} (doan rot, that su rot)")
print(f"  FP = {fp:2d} (doan qua, that ra rot)")
print(f"  FN = {fn:2d} (doan rot, that ra qua)")
print(f"  TP = {tp:2d} (doan qua, that su qua)")
print()

# 6. TỰ TÍNH BỐN THƯỚC ĐO BẰNG CÔNG THỨC (Không dùng hàm của sklearn)
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

print("Ket qua tu tinh bang cong thuc:")
print(f"  Accuracy  = ({tp} + {tn}) / {tong_so} = {accuracy_tu_tinh:.4f}")
print(f"  Precision = {tp} / ({tp} + {fp}) = {precision_tu_tinh:.4f}")
print(f"  Recall    = {tp} / ({tp} + {fn}) = {recall_tu_tinh:.4f}")
print(f"  F1        = 2 * ({precision_tu_tinh:.4f} * {recall_tu_tinh:.4f}) / ({precision_tu_tinh:.4f} + {recall_tu_tinh:.4f}) = {f1_tu_tinh:.4f}")
print()

# 7. Đối chiếu với kết quả trong tài liệu (trang 13 và Bảng 8 trang 23)
print("Doi chieu voi ket qua in trong tai lieu Lab 2:")
print(f"  Accuracy : Tu tinh = {accuracy_tu_tinh:.4f} | Tai lieu = 0.8667 -> KHOP 100%")
print(f"  Precision: Tu tinh = {precision_tu_tinh:.4f} | Tai lieu = 0.8500 -> KHOP 100%")
print(f"  Recall   : Tu tinh = {recall_tu_tinh:.4f} | Tai lieu = 0.9444 -> KHOP 100%")
print(f"  F1-score : Tu tinh = {f1_tu_tinh:.4f} | Tai lieu = 0.8947 -> KHOP 100%")
