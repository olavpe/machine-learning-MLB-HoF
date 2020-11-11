import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras_pandas.Automater import Automater

### Helper functions
def plot_roc_curve(fper, tper, name):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    file_name = '../result/ROC_curve_' + name + '.png'
    plt.savefig(file_name)
    plt.show()

### Importing the data

train_df = pd.read_csv('../data/train_data.csv', index_col=False)
test_df = pd.read_csv('../data/test_data.csv', index_col=False)
data_type_dict = {'numerical': [ 'G_all', 'finalGame', 'OPS', 'Years_Played',
                                 'Most Valuable Player', 'AS_games', 'Gold Glove',
                                 'Rookie of the Year', 'World Series MVP', 'Silver Slugger'],
                  'categorical': ['HoF']}


### Removing the answers for the input data
train_X_raw = train_df.drop(columns=['HoF'])
train_y_raw = train_df['HoF']
test_X_raw = test_df.drop(columns=['HoF'])
test_y_raw = test_df['HoF']

### Converting pandas arrays to numpy arrays
train_X = train_X_raw.to_numpy()
test_X = test_X_raw.to_numpy()
# print("train_X: ", train_X)
# print("test_X: ", test_X)

### Creating the label data for the train and test sets
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y_raw)
test_y = encoder.fit_transform(test_y_raw)

### Creating and compiling the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Training the model
# model.fit(train_X, train_y, epochs = 5, batch_size=1, validation_split=.2)
model.load_weights('model.h5')
# Testing the model
model.evaluate(test_X, test_y, verbose = 2)
full_predictions = model.predict_classes(test_X)

### Examining metrics
print('number of HoF players in data_set: ', test_y.sum())
tn, fp, fn, tp = confusion_matrix(test_y, full_predictions).ravel()
confusion_metrics = [tn, fp, fn, tp]
confusion_label = ["tn", "fp", "fn", "tp"]
for i in range(0,len(confusion_metrics)):
    print(confusion_label[i], ': ', confusion_metrics[i])

# full_predictions = full_predictions[:,1]
fper, tper, thresholds = roc_curve(test_y, full_predictions)
print("fper: ", fper)
print("tper: ", tper)
plot_roc_curve(fper, tper, "intial_results")
# small_predictions = model.predict(test_X[94:104])
# print(np.argmax(small_predictions, axis=1)) # [7, 2, 1, 0, 4]

# Saving the model
model.save_weights('model.h5')

