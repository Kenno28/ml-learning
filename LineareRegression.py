import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. Datensatz laden und in X, y aufteilen
data = load_diabetes()
X, y = data.data, data.target

# 2. Daten aufteilen in Training und Test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=40)

# 3. Lineares Regressionsmodell erstellen und trainieren
lm = LinearRegression()
lm.fit(X_train, y_train)

# 4. Vorhersagen auf Testdaten machen
predict = lm.predict(X_test)

# 5. Modell bewerten
print("MAE: ", metrics.mean_absolute_error(y_test, predict))
print("MSE: ", metrics.mean_squared_error(y_test, predict))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predict)))
print("Koeffizienten: ", lm.coef_)
print("Intercept: ", lm.intercept_)

# 6. Visualisierung der Residuen und der Vorhersagen
sns.displot((y_test - predict), kde=True)
plt.scatter(y_test, predict)
plt.xlabel("Tatsächliche Werte (y_test)")
plt.ylabel("Vorhersagen (predict)")
plt.title("Tatsächliche Werte vs. Vorhersagen")
plt.show()


