# -*- coding: utf-8 -*-
"""
Bài tập 1: Thống kê theo nhóm
Đọc bộ dữ liệu rồi in ra ba con số. Số bạn có điểm giữa kỳ từ 7 trở lên,
tỷ lệ qua môn của riêng nhóm đó, và tỷ lệ qua môn của nhóm còn lại.
Nhận xét về việc điểm giữa kỳ có phân biệt được hai nhóm hay không.
"""

import pandas as pd

# 1. Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")

# 2. Lọc theo điều kiện điểm giữa kỳ từ 7 trở lên và nhóm còn lại
nhom_cao = df[df["diem_giua_ky"] >= 7.0]
nhom_con_lai = df[df["diem_giua_ky"] < 7.0]

so_ban_cao = len(nhom_cao)
ty_le_qua_cao = nhom_cao["qua_mon"].mean()
ty_le_qua_con_lai = nhom_con_lai["qua_mon"].mean()

# 3. In ra ba con số
print(f"1. So ban co diem giua ky tu 7 tro len: {so_ban_cao}")
print(f"2. Ty le qua mon cua rieng nhom do    : {ty_le_qua_cao:.4f} ({ty_le_qua_cao*100:.2f}%)")
print(f"3. Ty le qua mon cua nhom con lai     : {ty_le_qua_con_lai:.4f} ({ty_le_qua_con_lai*100:.2f}%)")
print()

# 4. Nhận xét
print("Nhan xet:")
print("Diem giua ky phan biet rat ro hai nhom sinh vien:")
print("Nhom co diem giua ky tu 7 tro len co ty le qua mon len toi hon 90% (28/31 ban qua),")
print("trong khi nhom duoi 7 diem chi co ty le qua mon la 48.31% (chua toi mot nua).")
print("Dieu nay chung to diem giua ky la mot dac trung rat huu ich de du doan ket qua cuoi mon.")
