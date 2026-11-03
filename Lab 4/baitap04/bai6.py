# -*- coding: utf-8 -*-
"""Bài tập 6: Đo thời gian dự đoán.

Yêu cầu:
1. Tạo bộ dữ liệu ngẫu nhiên có 1000, 10000 và 100000 mẫu với 2 đặc trưng bằng np.random.default_rng(0).random.
2. Với mỗi cỡ, đo thời gian .fit và thời gian .predict cho 1000 mẫu mới bằng mô đun time.
3. Lập bảng kết quả rồi giải thích vì sao KNN được gọi là mô hình học lười.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Khởi tạo bộ sinh số ngẫu nhiên theo đề bài
rng = np.random.default_rng(0)

# 1000 mẫu mới dùng chung cho việc dự đoán ở các cỡ dữ liệu
so_mau_moi = 1000
X_moi = rng.random((so_mau_moi, 2))

cac_co_mau = [1_000, 10_000, 100_000]
bang_ket_qua = []

print("=" * 65)
print("KET QUA BAI TAP 6: DO THOI GIAN FIT VA PREDICT CUA KNN")
print("=" * 65)

for n in cac_co_mau:
    # Tạo tập dữ liệu huấn luyện kích thước n x 2
    X_hoc = rng.random((n, 2))
    y_hoc = rng.integers(0, 2, size=n)

    # 1. Đo thời gian .fit
    t0 = time.perf_counter()
    mo_hinh = KNeighborsClassifier(n_neighbors=5)
    mo_hinh.fit(X_hoc, y_hoc)
    t_fit = time.perf_counter() - t0

    # 2. Đo thời gian .predict cho 1000 mẫu mới
    t1 = time.perf_counter()
    mo_hinh.predict(X_moi)
    t_predict = time.perf_counter() - t1

    bang_ket_qua.append({
        "So mau hoc (N)": f"{n:,}",
        "Thoi gian .fit (giay)": f"{t_fit:.6f}",
        "Thoi gian .fit (ms)": f"{t_fit * 1000:.3f}",
        "Thoi gian .predict 1000 mau (giay)": f"{t_predict:.6f}",
        "Thoi gian .predict 1000 mau (ms)": f"{t_predict * 1000:.3f}",
    })

# Hiển thị bảng kết quả
df_ket_qua = pd.DataFrame(bang_ket_qua)
print(df_ket_qua.to_string(index=False))
print()

print("GIAI THICH VI SAO KNN DUOC GOI LA 'MO HINH HOC LUOI' (LAZY LEARNING):")
print("-" * 65)
print("1. Giai doan huan luyen (.fit) 'khong hoc gi ca':")
print("   - O cac mo hinh tham so nhu Hoi quy Logistic hay Mang No-ron, ham .fit phai chay")
print("     hang tram/hang nghin vong lap Gradient Descent de tinh toan, toi uu hoa bo tham so (w, b).")
print("   - Trong khi do voi KNN, buoc .fit chi don thuan la cat/luu toan bo du lieu hoc vao bo nho")
print("     (hoac dung cau truc cay tim kiem KD-Tree/Ball-Tree). Mo hinh khong tong quat hoa hay")
print("     truu tuong hoa du lieu thanh bat ky phuong trinh ranh gioi nao. Vi the no duoc goi la 'luoi hoc'.")
print()
print("2. Giai doan du doan (.predict) ganh toan bo khoi luong tinh toan:")
print("   - Khi co mau moi, KNN moi thuc su 'lam viec': phai tinh khoang cach tu mau do toi")
print("     hang ngan hang van mau trong bo nho roi moi sap xep tim ra k lang gieng gan nhat.")
print("   - Khi kich thuoc tap du lieu N tang len, tai nguyen va thoi gian tim kiem se tang vot.")
print("     Dieu nay hoan toan trai nguoc voi cac mo hinh eager learning (nhu Logistic/Linear Regression,")
print("     sau khi hoc xong chi can nhan ma tran y = w^T * x rat nhanh).")
print()
print("3. He qua thuc te:")
print("   - KNN huan luyen cuc ky nhanh (gan nhu tuc thi), nhung du doan cham va chiem nhieu RAM.")
print("   - Cang nhieu du lieu va cang nhieu chieu, KNN cang cham va de bi roi vao loi nguyen so chieu.")
print("   - Do do KNN rat thich hop voi du lieu nho den vua, it chieu va can cap nhat du lieu moi lien tuc.")
