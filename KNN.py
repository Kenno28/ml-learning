import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot
from sklearn.model_selection  import GridSearchCV

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Daten flach machen von 3D -> 2D
x_train_flat = x_train.reshape(x_train.shape[0], -1)  # (60000, 784)
x_test_flat = x_test.reshape(x_test.shape[0], -1)    # (10000, 784)

scaler = StandardScaler()

scaled_X = scaler.fit_transform(x_train_flat)
test_scaled_X = scaler.transform(x_test_flat)

param_grid= {"n_neighbors":[1,5,6,8,10]}

#Um den besten Hyper Parameter Werte zu finden
grid = GridSearchCV(KNeighborsClassifier(),param_grid,verbose=1)

grid.fit(scaled_X,y_train)

predict = grid.predict(test_scaled_X)


print(confusion_matrix(y_test,predict))
#Result
#[[ 965    1    0    3    1    4    5    0    1    0]
# [   0 1127    3    1    1    0    3    0    0    0]
# [  10    7  966   19    4    0    3   10    9    4]
# [   0    2    3  949    2   19    1   11   16    7]
# [   1    9    5    1  920    2    5    6    3   30]
# [   5    0    1   30    4  813   12    1   18    8]
# [  13    4    1    2    4    5  929    0    0    0]
# [   0   14   11    4    6    1    0  959    1   32]
# [  14    2    7   20    8   30    3    8  875    7]
# [   4    4    6    9   15    6    0   30    4  931]]

print(classification_report(y_test,predict))
#Result
#             precision    recall  f1-score   support
#
#           0       0.95      0.98      0.97       980
#           1       0.96      0.99      0.98      1135
#           2       0.96      0.94      0.95      1032
#           3       0.91      0.94      0.93      1010
#           4       0.95      0.94      0.95       982
#           5       0.92      0.91      0.92       892
#           6       0.97      0.97      0.97       958
#           7       0.94      0.93      0.93      1028
#           8       0.94      0.90      0.92       974
#           9       0.91      0.92      0.92      1009
#
#   accuracy                           0.94     10000
#   macro avg       0.94      0.94      0.94     10000
#weighted avg       0.94      0.94      0.94     10000