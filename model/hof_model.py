import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras_pandas.Automater import Automater

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
print("train_X: ", train_X)
print("test_X: ", test_X)

### Creating the label data for the train and test sets
encoder = LabelEncoder()
encoder.fit(train_y_raw)
train_y = encoder.transform(train_y_raw)
print("train_y: ", train_y)
encoder = LabelEncoder()
encoder.fit(test_y_raw)
test_y = encoder.transform(test_y_raw)

# train_X = []
# # train_y = []
# for index, row in train_X_raw.iterrows():
#     train_X.append(np.array([row]))
# # print("train_X: ", train_X)

# test_X = []
# # test_y = []
# for index, row in test_X_raw.iterrows():
#     test_X.append(np.array([row]))
# print("test_X: ", test_X)

# encoder = LabelEncoder()
# encoder.fit(train_)

### Selecting the data to be used
# train_X, train_y = auto.transform(train_df)
# test_X, test_y = auto.transform(test_df, df_out=False)
# print('train_X: ', train_X)

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
model.fit(train_X, train_y, epochs = 5, batch_size=1, validation_split=.2)
# Testing the model
model.evaluate(test_X, test_y, verbose = 2)
predictions = model.predict(test_X[94:104])

print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Saving the model
model.save_weights('model.h5')

