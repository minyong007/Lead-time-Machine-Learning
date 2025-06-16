import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

#  Dummy
df = pd.read_csv('data_barang.csv', sep=';')
df.columns = df.columns.str.strip()
print("Kolom:", df.columns.tolist())

#  Hitung lead_time (hari)
df['tanggal_order'] = pd.to_datetime(df['tanggal_order'], dayfirst=True)
df['tanggal_terima'] = pd.to_datetime(df['tanggal_terima'], dayfirst=True)
df['lead_time'] = (df['tanggal_terima'] - df['tanggal_order']).dt.days

#  Encoding otomatis fitur kategorikal
label_encoders = {}
for col in ['supplier', 'lokasi', 'jenis_barang']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Mapping {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

#  Split data
X = df[['supplier', 'lokasi', 'jenis_barang', 'jumlah_order']]
y = df['lead_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

#  Evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

#  Prediksi contoh baru (update sesuai kategori di data barumu)
try:
    contoh_baru = pd.DataFrame({
        'supplier': [label_encoders['supplier'].transform(['Karunia Baru'])[0]],
        'lokasi': [label_encoders['lokasi'].transform(['Palangka Raya'])[0]],
        'jenis_barang': [label_encoders['jenis_barang'].transform(['Freon R32'])[0]],
        'jumlah_order': [120]
    })
    prediksi_lead_time = model.predict(contoh_baru)
    print(f'Prediksi waktu tunggu (hari): {prediksi_lead_time[0]:.2f}')
except Exception as e:
    print("Error saat prediksi data baru:", e)


joblib.dump(model, 'leadtime_model_linear.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

#  Plot distribusi Lead Time
plt.figure(figsize=(6,4))
plt.hist(df['lead_time'], bins=range(int(df['lead_time'].min()), int(df['lead_time'].max())+2), color='skyblue', edgecolor='black')
plt.title('Distribusi Lead Time Pemasok')
plt.xlabel('Lead Time (hari)')
plt.ylabel('Jumlah Order')
plt.tight_layout()
plt.show()

# Plot rata-rata Lead Time per Supplier
supplier_map_inv = {idx: name for idx, name in enumerate(label_encoders['supplier'].classes_)}
supplier_leadtime = df.groupby('supplier')['lead_time'].mean()
plt.figure(figsize=(5,4))
plt.bar([supplier_map_inv[s] for s in supplier_leadtime.index], supplier_leadtime, color='salmon')
plt.title('Rata-rata Lead Time per Supplier')
plt.xlabel('Supplier')
plt.ylabel('Rata-rata Lead Time (hari)')
plt.tight_layout()
plt.show()

#  Scatter plot Actual vs Predicted
plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred, color='skyblue', alpha=0.7)
plt.xlabel('Actual Lead Time')
plt.ylabel('Predicted Lead Time')
plt.title('Actual vs Predicted Lead Time')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
plt.tight_layout()
plt.show()