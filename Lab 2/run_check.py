import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/sinh_vien.csv")

print("--- KIEM TRA DU LIEU GOC ---")
print("Shape:", df.shape)
print("Mean qua_mon:", df["qua_mon"].mean())
print("Mean gio_on theo nhom:")
print(df.groupby("qua_mon")["gio_on"].mean())

print("\n--- BAI TAP 1 ---")
nhom_cao = df[df["diem_giua_ky"] >= 7.0]
nhom_con_lai = df[df["diem_giua_ky"] < 7.0]
so_cao = len(nhom_cao)
ty_le_cao = nhom_cao["qua_mon"].mean()
ty_le_con_lai = nhom_con_lai["qua_mon"].mean()
print(f"So ban diem giua ky >= 7: {so_cao}")
print(f"Ty le qua mon nhom >= 7: {ty_le_cao:.4f}")
print(f"Ty le qua mon nhom con lai: {ty_le_con_lai:.4f}")

print("\n--- BAI TAP 3 ---")
X = df[["gio_on"]]
y = df["qua_mon"]
model = LogisticRegression()
model.fit(X, y)
w = float(model.coef_[0][0])
b = float(model.intercept_[0])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
for gio in [3, 8, 12.89, 18, 26]:
    z = w * gio + b
    p = sigmoid(z)
    nhan = 1 if p >= 0.5 else 0
    print(f"Gio: {gio:5.2f}, z: {z:8.4f}, prob: {p:6.4f}, nhan: {nhan}")

print("\n--- BAI TAP 4 ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

print("\n--- BAI TAP 5 ---")
p_test = model.predict_proba(X_test)[:, 1]
best_f1 = -1
best_thresh = -1
for thresh in np.arange(0.05, 1.0, 0.05):
    yp = (p_test >= thresh).astype(int)
    f = f1_score(y_test, yp, zero_division=0)
    print(f"Thresh: {thresh:.2f}, F1: {f:.4f}")
    if f > best_f1:
        best_f1 = f
        best_thresh = thresh
print(f"Best thresh: {best_thresh:.2f} with F1: {best_f1:.4f}")

print("\n--- BAI TAP 6 ---")
prec_0 = precision_score(y_test, y_pred, pos_label=0)
rec_0 = recall_score(y_test, y_pred, pos_label=0)
f1_0 = f1_score(y_test, y_pred, pos_label=0)
print(f"Lop 0 (Rot mon): Precision={prec_0:.4f}, Recall={rec_0:.4f}, F1={f1_0:.4f}")
