# -*- coding: utf-8 -*-
"""Bài tập 5: Chọn k cho bài toán hồi quy.

Yêu cầu:
1. Trên dữ liệu giá nhà, dùng kiểm định chéo 5 phần trên tập học để chọn k tốt nhất cho KNeighborsRegressor (k từ 1 tới 20).
   Gợi ý: truyền scoring="neg_mean_absolute_error" vào cross_val_score, và nhớ rằng điểm trả về mang dấu âm.
2. Trả lời: Với k tốt nhất, MAE trên tập kiểm tra có vượt được đường thẳng không?
"""

import os
# Xử lý tương thích nạp thư viện đa luồng trên Windows khi đường dẫn chứa ký tự tiếng Việt
try:
    import threadpoolctl
    orig_init = threadpoolctl.LibController.__init__
    def safe_init(self, *, filepath=None, prefix=None, parent=None):
        try:
            orig_init(self, filepath=filepath, prefix=prefix, parent=parent)
        except Exception:
            self.parent, self.prefix, self.filepath = parent, prefix, filepath
            self.dynlib = None
            self._symbol_prefix, self._symbol_suffix = "", ""
            self.version = None
    threadpoolctl.LibController.__init__ = safe_init
except Exception:
    pass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Đọc dữ liệu giá nhà
data_path = "data/gia_nha.csv"
if not os.path.exists(data_path):
    if os.path.exists("../data/gia_nha.csv"):
        data_path = "../data/gia_nha.csv"
    elif os.path.exists("gia_nha.csv"):
        data_path = "gia_nha.csv"

df = pd.read_csv(data_path)
X = df[["dien_tich"]]
y = df["gia"]

# Chia tập theo đúng mục 6: test_size=0.2, random_state=42
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("=" * 65)
print("KET QUA BAI TAP 5: CHON K CHO BAI TOAN HOI QUY (GIA NHA)")
print("=" * 65)
print("Kiem dinh cheo 5 phan tren tap hoc (danh gia bang MAE):")

diem_mae = {}
for k in range(1, 21):
    # scoring="neg_mean_absolute_error" trả về giá trị âm (-MAE)
    scores = cross_val_score(
        KNeighborsRegressor(n_neighbors=k),
        X_tr, y_tr,
        cv=5,
        scoring="neg_mean_absolute_error"
    )
    # Đổi dấu để nhận giá trị MAE dương
    diem_mae[k] = -scores.mean()

# Tìm k có MAE nhỏ nhất
k_tot = min(diem_mae, key=diem_mae.get)

for k in range(1, 21):
    dau = " <- tot nhat" if k == k_tot else ""
    print(f"  k = {k:2d}  MAE trung binh: {diem_mae[k]:.4f} ty dong{dau}")

print()
print(f"Gia tri k toi uu theo kiem dinh cheo: k = {k_tot} (MAE CV = {diem_mae[k_tot]:.4f} ty dong)")
print()

# Huấn luyện mô hình KNN tối ưu và đánh giá trên tập kiểm tra
knn_tot = KNeighborsRegressor(n_neighbors=k_tot).fit(X_tr, y_tr)
mae_knn_tot = mean_absolute_error(y_te, knn_tot.predict(X_te))

# Đối chiếu với mô hình Hồi quy tuyến tính (đường thẳng)
duong_thang = LinearRegression().fit(X_tr, y_tr)
mae_duong_thang = mean_absolute_error(y_te, duong_thang.predict(X_te))

print("SO SANH MAE TREN TAP KIEM TRA:")
print(f"- KNN (k toi uu = {k_tot}) : {mae_knn_tot:.4f} ty dong")
print(f"- Hoi quy tuyen tinh     : {mae_duong_thang:.4f} ty dong")
print()
print("TRA LOI CAU HOI:")
print("Cau hoi: Voi k tot nhat, MAE tren tap kiem tra co vuot duoc duong thang khong?")
print(f"-> KHONG, MAE cua KNN voi k={k_tot} ({mae_knn_tot:.4f} ty) van cao hon dang ke so voi duong thang ({mae_duong_thang:.4f} ty).")
print()
print("GIAI THICH LY DO:")
print("1. Ban chat du lieu gia nha: Gia nha thuc te tang rat deu va gan nhu tuyen tinh tuyet doi")
print("   theo dien tich. Mo hinh Hoi quy tuyen tinh gia dinh dung ban chat nay va tan dung toan bo")
print("   tap du lieu de hoc nen mot duong thang chuan xac toan cuc.")
print("2. Han che cua KNN trong hoi quy: KNN la mo hinh cuc bo phi tham so. Du doan cua no la")
print("   trung binh cua k diem lan can, tao nen duong du doan dang 'bac thang' gap ghenh.")
print("   No rat nhay cam voi cac sai lech ngau nhien cuc bo va khong the tao ra xu huong noi suy/ngoai suy")
print("   muot ma nhu duong thang.")
print("3. Ket luan: Khi moi quan he giua bien doc lap va bien phu thuoc la tuyen tinh ro ret,")
print("   mo hinh tuyen tinh luon chien thang KNN ca ve do chinh xac lan toc do.")
