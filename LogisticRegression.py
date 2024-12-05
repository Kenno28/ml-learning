import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Diabetes-Datensatz laden
df = load_diabetes()

# DataFrame erstellen: Features und Zielwerte
df2 = pd.DataFrame(df.data, columns=df.feature_names)  # Features hinzufügen
df2['target'] = df.target  # Zielwert hinzufügen

# Features und Zielwert definieren
X = df2.drop("target", axis=1)  # Features (Unabhängige Variablen)
y = df2["target"]  # Zielwert (Kontinuierlich)

# Zielwert binarisieren, um ihn für die Klassifikation nutzbar zu machen
threshold = np.median(y)  # Median als Schwellenwert für die Binarisierung
y_binary = (y > threshold).astype(int)  # Klassen: 1 (über Median) und 0 (unter Median)

# Trainings- und Testdaten aufteilen (40 % Testdaten, zufällige Aufteilung mit Seed)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.4, random_state=101
)

# Logistic Regression-Modell erstellen und konfigurieren
lg = LogisticRegression(max_iter=500)  # Erhöhung der Iterationsanzahl für Konvergenz

# Modell trainieren
try:
    lg.fit(X_train, y_train)
except Exception as e:
    print(f"Fehler beim Training des Modells: {e}")

# Vorhersagen auf den Testdaten erstellen
prediction = lg.predict(X_test)

# Klassifikationsbericht und Konfusionsmatrix ausgeben
print("Klassifikationsbericht:")
print(classification_report(y_test, prediction))
print("Konfusionsmatrix:")
print(confusion_matrix(y_test, prediction))