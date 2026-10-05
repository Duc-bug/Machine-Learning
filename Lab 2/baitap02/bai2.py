# -*- coding: utf-8 -*-
"""
Bài tập 2: Vẽ hàm sigmoid
Vẽ đồ thị hàm sigmoid với z chạy từ -8 tới 8, lưu thành tệp sigmoid.png.
Trên cùng hình, vẽ thêm một đường ngang tại mức 0.5 và một đường dọc tại z = 0.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 1. Tạo mảng z từ -8 tới 8
z = np.linspace(-8, 8, 400)
sigma_z = sigmoid(z)

# 2. Vẽ đồ thị
plt.figure(figsize=(8, 5))
plt.plot(z, sigma_z, color="#1f77b4", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")

# Đường ngang tại mức 0.5 và đường dọc tại z = 0
plt.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label=r"Đường ngang $\sigma(z) = 0.5$")
plt.axvline(0, color="gray", linestyle=":", linewidth=1.2, label=r"Đường dọc $z = 0$")

# Điểm giao tại gốc (0, 0.5)
plt.scatter([0], [0.5], color="red", s=60, zorder=5)
plt.annotate(r"$\sigma(0) = 0.5$", xy=(0, 0.5), xytext=(0.8, 0.42),
             arrowprops=dict(arrowstyle="->", color="red", lw=1),
             fontsize=11, color="red")

plt.title("Đồ thị hàm Sigmoid", fontsize=14, fontweight="bold")
plt.xlabel("z", fontsize=12)
plt.ylabel(r"$\sigma(z)$", fontsize=12)
plt.xlim(-8, 8)
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left", fontsize=10)

# 3. Lưu thành tệp sigmoid.png (ở thư mục gốc và thư mục baitap02 để tiện sử dụng)
plt.savefig("baitap02/sigmoid.png", dpi=150, bbox_inches="tight")
plt.savefig("sigmoid.png", dpi=150, bbox_inches="tight")
plt.close()

print("Da ve va luu thanh cong tep sigmoid.png!")
