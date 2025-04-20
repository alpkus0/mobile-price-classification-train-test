import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Random Forest → Birden fazla karar ağacı (daha güçlü model)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- DATAMIZI TANIYALIM ---
# Eğitim verisini oku
train_df = pd.read_csv(r"C:\Users\alper\Masaüstü\Masaüstü\ACUN MEDYA VERİ BİLİMİ"
                       r"\ödevler\10.hafta ödev\archive\train.csv")

# Test verisini oku
test_df = pd.read_csv(r"C:\Users\alper\Masaüstü\Masaüstü\ACUN MEDYA VERİ BİLİMİ\ödevler\10.hafta ödev\archive\test.csv")

# Veriyi inceleyelim
print("\n", " " * 30, "İlk 5 veriye bakalım\n")
print(train_df.head())

print("\n", ("-" * 150), "\n")

# Satır, sutün sayısını öğrenelim
print(" " * 30, "Satır, Sutün Sayısı\n")
print(train_df.shape)

print("\n", ("-" * 150), "\n")

# Verinin yapısını tanıyalım
print(" " * 30, "Veri yapısı\n")
print(train_df.info())

print("\n", ("-" * 150), "\n")

# Eksik var mı yok mu kontrol edelim
print(" " * 30, "Eksik veri kontrolü\n")
print(train_df.isnull().sum())

# --- HEDEFLERİ BELİRLEME ---
x = train_df.drop("price_range", axis=1)  # Bağımsız Değişkenler (Girdiler: battery_power, dual_sim vs.)
y = train_df["price_range"]  # Hedef Değişken (price_range)

# --- EĞİTİM & TEST SETİNE AYIRMA ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- RANDOM FOREST MODEL KULLANIMI ---
random_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Model parametreleri
random_model.fit(x_train, y_train)
random_pred = random_model.predict(x_test)

random_accuracy = accuracy_score(y_test, random_pred)
print("\n", ("-" * 150), "\n")
print(f"Random Forest Doğruluğu: {random_accuracy:.4f}\n")

target_names = ["Ekonomik Segment", "Orta Segment", "Üst Segment", "Lüks Segment"]

# --- CLASSIFICATION REPORT ---
report = classification_report(y_test, random_pred, target_names=target_names)
print("\n", ("-" * 150), "\n")
print("\nRandom Forest - Sınıflandırma Raporu:\n")
print(report)

# --- KARIŞIKLIK MATRİSİ (CONFUSION MATRIX) ---
conf_mat = confusion_matrix(y_test, random_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=target_names)
disp.plot(cmap=plt.cm.Oranges)
plt.title("Random Forest - Confusion Matrix")
plt.show()
