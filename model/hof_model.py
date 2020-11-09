import pandas as pd
import numpy as np
# import mnist
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras_pandas.Automater import Automater

### Importing the data

train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')
data_type_dict = {'numerical': [ 'G_all', 'finalGame', 'OPS', 'Years_Played',
                                 'Most Valuable Player', 'AS_games', 'Gold Glove',
                                 'Rookie of the Year', 'World Series MVP', 'Silver Slugger'],
                  'categorical': ['HoF']}
# train_X = np.empty([0])
# train_y = np.empty([0])
# for index, row in train_df.iterrows():
#     # train_X = np.array([train_X, row])
#     train_X = np.array([train_X, row])



# encoder = LabelEncoder()
# encoder.fit(train_)

### Selecting the data to be used
auto = Automater(data_type_dict=data_type_dict, output_var='HoF')
auto.fit(train_df)
# train_X, train_y = auto.fit_transform(train_df)
train_X, train_y = auto.transform(train_df)
test_X, test_y = auto.transform(test_df, df_out=False)
print('train_X: ', train_X)
# print('train_X: ', train_X)
# print('train_y: ', train_y.head())
# print('train_y: ', train_y.shape)
# print('test_X: ', test_X.shape)
# print('test_y: ', test_y.shape)
## Testing with Keras-pandas
x = auto.input_nub # Starting model with input nub
print('first input nub: ', x)
# x = Dense(10, activation='relu', input_shape=(10,))(x)
# x = Dense(32, activation='relu')(x)
# x = auto.output_nub(x, activation='sigmoid')
x = auto.output_nub(x)
print('output nub: ', x)

model = Model(inputs=auto.input_layers, outputs=x)
model.compile(
    optimizer='adam',
    loss=auto.suggest_loss(),
    # loss='binary_crossentropy',
    metrics=['accuracy']
)
# Training the model
model.fit(train_X, train_y, epochs = 5, batch_size = 32, validation_split=.2)
# Testing the model
model.evaluate(test_X, test_y, verbose = 2)
# Saving the model
model.save_weights('model.h5')





# # # The first time you run this might be a bit slow, since the
# # # mnist package has to download and cache the data.
# # train_images = mnist.train_images()
# # train_labels = mnist.train_labels()
# # test_images = mnist.test_images()
# # test_labels = mnist.test_labels()


# # model = Sequential([
# #     Dense(10, activation='relu', input_shape=(10,)),
# #     Dense(32, activation='relu'),
# #     Dense(1, activation='sigmoid'),
# # ])

# # model.compile(
# #     optimizer='adam',
# #     loss='binary_crossentropy',
# #     metrics=['accuracy']
# # )

# print('train_X length: ', len(train_X))
# print('train_X[0] length: ', len(train_X[0]))
# # print('train_X[0][0] length: ', len(train_X[0][0]))
# print('test_X length: ', len(test_X))
# print('test_X[0] length: ', len(test_X[0]))
# # print('test_X[0][0] length: ', len(test_X[0][0]))
# # print(test_X.shape) # (60000, 28, 28)
# # tf.print("Test X: ", test_X) # (60000,)
# # print(train_y.shape) # (60000,)
# # tf.print("Train y: ", train_y) # (60000,)
# # test_cat = to_categorical(test_y)
# # print("testing to_categorical", test_cat)


# # model.fit(
# #     train_data,
