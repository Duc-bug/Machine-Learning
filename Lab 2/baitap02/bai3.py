# -*- coding: utf-8 -*-
"""
Bài tập 3: Dự đoán cho một bạn cụ thể
Khớp lại mô hình một biến như ở mục 4, rồi viết một hàm du_doan(gio)
nhận vào số giờ ôn và in ra ba thứ: Giá trị z, xác suất qua môn, và nhãn theo ngưỡng 0.5.
Gọi hàm đó với các giá trị 3, 8, 12.89, 18 và 26 giờ.
Giải thích vì sao với 12.89 giờ thì xác suất ra gần đúng 0.5.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Đọc dữ liệu và khớp mô hình một biến gio_on
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
    print(f"Gio on: {gio:5.2f}h | z = {z:7.4f} | Xac suat qua = {xac_suat:.4f} | Nhan: {nhan} ({'Qua' if nhan==1 else 'Rot'})")
    return z, xac_suat, nhan


print(f"He so goc w = {w:.6f}, He so chan b = {b:.6f}")
print("Moc 50/50 theo ly thuyet: -b/w =", f"{-b/w:.4f} gio\n")

print("Ket qua du doan cho cac muc gio:")
for g in [3, 8, 12.89, 18, 26]:
    du_doan(g)

print()
print("Giai thich vi sao voi 12.89 gio thi xac suat ra gan dung 0.5:")
print("1. Ham sigmoid co tinh chat: sigma(z) = 0.5 khi va chi khi z = 0.")
print("2. Phan tuyen tinh z duoc tinh bang cong thuc: z = w * gio_on + b.")
print("   De z = 0 thi gio_on = -b / w = -(-5.064890) / 0.392819 = 12.8937 gio.")
print("3. Khi lay gio_on = 12.89, ta co:")
print(f"   z = {w:.6f} * 12.89 + ({b:.6f}) = {w * 12.89 + b:.6f} (rat sat 0).")
print(f"   Do do, sigma(z) = sigma({w * 12.89 + b:.6f}) = {sigmoid(w * 12.89 + b):.4f} ~ 0.5000.")
