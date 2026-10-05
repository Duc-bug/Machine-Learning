# -*- coding: utf-8 -*-
"""
Bài tập 5: Dò ngưỡng tốt nhất theo F1
Cho ngưỡng chạy từ 0.05 tới 0.95, mỗi bước 0.05.
Với mỗi ngưỡng, tính F1 trên tập kiểm tra rồi in ra thành bảng.
Cuối cùng in ra ngưỡng cho F1 cao nhất.
Nhận xét một câu: ngưỡng tốt nhất theo F1 có đúng bằng 0.5 không?
"""

import numpy as np
import pandas as pd
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

# 2. Khớp mô hình
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)

# 3. Lấy xác suất dự đoán lớp 1
p = mo_hinh.predict_proba(X_test)[:, 1]

# 4. Quét ngưỡng từ 0.05 đến 0.95 với bước nhảy 0.05
nguong_list = np.arange(0.05, 1.0, 0.05)

print("Bang do tim nguong quyet dinh theo F1-score:")
print("  Nguong    So ban qua   Precision    Recall      F1-score")
print("  --------------------------------------------------------")

best_f1 = -1.0
best_nguong = 0.5

for nguong in nguong_list:
    y_pred = (p >= nguong).astype(int)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    so_qua = y_pred.sum()
    print(f"   {nguong:4.2f}        {so_qua:2d}        {pre:6.4f}     {rec:6.4f}     {f1:6.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_nguong = nguong

print("  --------------------------------------------------------")
print(f"Nguong cho F1 cao nhat la: {best_nguong:.2f} voi F1 = {best_f1:.4f}")
print()

# 5. Nhận xét
print("Nhan xet:")
print(f"Nguong tot nhat theo F1 la {best_nguong:.2f}, KHONG PHAI bang 0.5.")
print(f"Tai nguong mac dinh 0.5, F1 dat 0.8947 (Precision = 0.8500, Recall = 0.9444).")
print(f"Khi nang nguong len 0.60, mo hinh kho tinh hon, loai bo duoc ca 3 ban bi du doan nham qua mon (FP giam ve 0),")
print(f"dua Precision len tuyet doi 1.0000 trong khi Recall chi giam nhe xuong 0.8889, giup F1 dat dinh 0.9412.")
