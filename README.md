```import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Veriyi oku
df = pd.read_csv("medical.zip")

# Kategorik değişkenleri dummy değişkenlere çevir
df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

# Tüm veriyi float'a çevir
df = df.astype(float)

# Bağımlı ve bağımsız değişkenleri ayır
y = df[["charges"]]
x = df.drop("charges", axis=1)

# Doğrusal regresyon modelini eğit
model = LinearRegression().fit(x, y)

# Modelin R^2 skoru
print("Model Skoru (R²):", model.score(x, y))

# Örnek bir tahmin
example = [[19, 33, 1, 0, 0, 1, 0, 1]]  # Bu örnek verinin sırayla tüm özelliklere uygun olması gerekir
prediction = model.predict(example)
print("Örnek Tahmin:", prediction)

# Tahminleri oluştur
df["Tahmin"] = model.predict(x)

# Farkları ve hata metriklerini hesapla
df["Fark"] = df["charges"] - df["Tahmin"]
df["Hatanın Karesi"] = df["Fark"] ** 2
df["Mutlak Fark"] = df["Fark"].abs()
df["Yuzdelik Hata"] = df["Mutlak Fark"] / df["charges"]

# Hata metriklerini yazdır
mse = mean_squared_error(df["charges"], df["Tahmin"])
mae = mean_absolute_error(df["charges"], df["Tahmin"])
mape = mean_absolute_percentage_error(df["charges"], df["Tahmin"])

print("Ortalama Kare Hata (MSE):", mse)
print("Ortalama Mutlak Hata (MAE):", mae)
print("Ortalama Mutlak Yüzde Hata (MAPE):", mape)```
