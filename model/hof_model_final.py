from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, auc
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve
import scikitplot as skplt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from keras_pandas.Automater import Automater


### ------------------ Setting the table ------------------ ###

### Hyper-parameters
HP = {
    'NAME': 'full',
    'INFO': 'checking_testing_final',
    'EPOCHS': 50,
    'FOLDS': 2,
    'BATCH_SIZE': 1,
    'OPTIMIZER': 'adam',
    'LOSS': 'binary_crossentropy',
    'METRICS': ['accuracy', 'Recall'],
    'DATASET': 'raw'
}

### Adding the information to the log file
with open("../result/master_log.txt", "a") as file:
    file.write("\n")
    file.write("\n")
    print(HP, file=file)

### Setting up the CSVlogger and Tensorboard
csv_logger = CSVLogger('../result/master_log.txt', append=True, separator=';')
log_dir = "../logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

### Helper functions
def plot_roc_curve(fper, tper, name):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    file_name = '../result/ROC_curve_' + name + datetime.now().strftime("%Y%m%d-%H%M%S") + '.png'
    plt.savefig(file_name)
    plt.show()

def save_model(model, name):
    # model_json = model.model.to_json()
    # model_name = "model_" + name +".json"
    # with open(model_name, "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    weights_name = "model_" + name +".h5"
    model.model.save_weights(weights_name)
    print("Saved model to disk")

def load_model(model, name):
    # json_name = "model_" + name + ".json"
    # json_file = open(json_name, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    # # load weights into new model
    model_name = "model_" + name + ".h5"
    model.load_weights(model_name)
    model.compile(optimizer= HP['OPTIMIZER'], loss= HP['LOSS'], metrics=HP['METRICS'])
    print("Loaded model from disk")
    return model

### ---------------- Importing the data ---------------- ###

train_df = pd.read_csv('../data/train_data_' + HP['NAME'] + '.csv', index_col=False)
test_df = pd.read_csv('../data/test_data_' + HP['NAME'] + '.csv', index_col=False)
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

### Creating the label data for the train and test sets
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y_raw)
test_y = encoder.fit_transform(test_y_raw)


### ---------------- Creating and compiling the model ---------------- ###

### Weighting the classes for bias datasets
# class_weights = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
class_weights = {0: 1.0, 1: 15.0}
### Setting up saving of the model weights
model_weights_name = HP['NAME'] + '_model.h5'
checkpointer = ModelCheckpoint(model_weights_name, monitor='Recall', verbose=0)
print("class weights: ", class_weights)
print("value counts of Y in train_y: ", train_y.sum())
print("value counts of N in train_y: ", len(train_y) - train_y.sum())

### Creating model
def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(10,)),
        Dense(5, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer= HP['OPTIMIZER'],
        loss= HP['LOSS'],
        metrics=HP['METRICS'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=HP['EPOCHS'],
                            batch_size=HP['BATCH_SIZE'], verbose = 2, )
model.fit(train_X, train_y, callbacks=[csv_logger, tensorboard_callback])


### ---------------- Saving the model ---------------- ###

### Saving Entire Model
# save_model(model, "test_final_check")
# model.model.save("model_test_final_different.h5")
### Loading Entire Model
# model = load_model(model, "test_final")
# model.load("model_test_final_different.h5")


### --------- Evaluating the model and generating metrics -------------- ###

# Testing the model
pred = model.predict(test_X)
y_score = model.predict_proba(test_X, batch_size=HP['BATCH_SIZE'])

# Calculating overall metrics
accuracy = accuracy_score(test_y, pred)
tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
confusion_mat = [tn, fp, fn, tp]
auroc = roc_auc_score(test_y, y_score[:,0])
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

# Showing numerical results
confusion_label = ["tn", "fp", "fn", "tp"]
for i in range(0,len(confusion_mat)):
    print(confusion_label[i], ': ', confusion_mat[i])
print("###### ---------Overall Results --------- ######")
print("accuracy: ", accuracy)
print("confusion_mat: ", confusion_mat)
print("auroc: ", auroc)
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)


### ---------------- Plotting and graphing the data ---------------- ###

# ROC curve
fper, tper, thresholds = roc_curve(test_y, y_score[:,1])
plot_roc_curve(fper, tper, HP['NAME'])
man_auroc = auc(fper, tper)
print("man_auroc: ", man_auroc)

# Generating a confusion matrix
skplt.metrics.plot_confusion_matrix(test_y, pred, normalize=True)
confusion_mat_string = "../result/confusion_mat_" + HP['NAME']+ datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
plt.savefig(confusion_mat_string)
plt.show()

# Generating precision-recall curve
skplt.metrics.plot_precision_recall(test_y, y_score)
precision_recall_curve_string = "../result/precision_recall_curve_" + HP['NAME'] + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
plt.savefig(precision_recall_curve_string)
plt.show()


### ---------------- Saving the metrics ---------------- ###

metric_dict = {
    'True Negative': tn,
    'True Positive': tp,
    'False Negative': fn,
    'False Positive': fp,
    'AUROC': auroc,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1': f1
}
with open("../result/master_log.txt", "a") as file:
    print(metric_dict, file=file)
