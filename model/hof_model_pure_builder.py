from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, recall_score
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve
import scikitplot as skplt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras_pandas.Automater import Automater


### ------------------ Setting the table ------------------ ###

### Hyper-parameters
HP = {
    'NAME': 'full',
    # 'INFO': 'weightedClass-7,3-layers-trying long run',
    'INFO': 'single layer, 15 class weights',
    'EPOCHS': 50,
    'FOLDS': 10,
    'BATCH_SIZE': 1,
    'OPTIMIZER': 'adam',
    'LOSS': 'binary_crossentropy',
    'METRICS': ['accuracy', 'Recall'],
    # 'METRICS': 'accuracy',
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
class_weights = {0: 1.0, 1: 17.0}
### Setting up saving of the model weights
model_weights_name = HP['NAME'] + '_model.h5'
checkpointer = ModelCheckpoint(model_weights_name, monitor='val_loss', verbose=0)
print("class weights: ", class_weights)
print("value counts of Y in train_y: ", train_y.sum())
print("value counts of N in train_y: ", len(train_y) - train_y.sum())

### Implementing Early Stopping
stop = EarlyStopping(monitor='Recall', mode='max', verbose=1, patience='10')

### Creating model
def create_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(10,)),
        # Dense(7, activation='relu'),
        Dense(5, activation='relu'),
        # Dense(32, activation='relu'),
        # Dense(3, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer= HP['OPTIMIZER'],
        loss= HP['LOSS'],
        # metrics=['accuracy', metrics.])
        metrics=HP['METRICS'])
    return model

kfold = StratifiedKFold(n_splits=HP['FOLDS'], shuffle=True)
cv_accuracy = []
cv_auroc = []
# cv_confunsion_mat = np.empty(0)
cv_confusion_mat = []

for train_index, val_index in kfold.split(train_X, train_y):
    model = KerasClassifier(build_fn=create_model, epochs=HP['EPOCHS'],
                                batch_size=HP['BATCH_SIZE'], verbose = 2)
    model.fit(train_X[train_index], train_y[train_index],
              callbacks=[csv_logger, tensorboard_callback])
    pred = model.predict(train_X[val_index])
    fold_accuracy = accuracy_score(train_y[val_index], pred)
    tn, fp, fn, tp = confusion_matrix(train_y[val_index], pred).ravel()
    confusion_mat_fold = [tn, fp, fn, tp]
    fold_auroc = roc_auc_score(train_y[val_index], pred)
    cv_auroc.append(fold_auroc)
    cv_accuracy.append(fold_accuracy)
    cv_confusion_mat.append(confusion_mat_fold)
    # cv_confunsion_mat = cv_confunsion_mat.append(confusion_mat_fold)
    # cv_confunsion_mat = np.vstack([cv_confunsion_mat, confusion_mat_fold])
    print("## -- Fold Results -- ##")
    print("accuracy: ", fold_accuracy)
    print("auroc: ", fold_auroc)
    print("confusion_mat: ", confusion_mat_fold)

# Calculating overall metrics
tot_accuracy = np.mean(cv_accuracy)
tot_auroc = np.mean(cv_auroc)
print("cv_confusion_mat_prestack: ", cv_confusion_mat)
cv_confusion_mat = np.stack(cv_confusion_mat, axis=0)
print("cv_confusion_mat: ", cv_confusion_mat)
tot_confusion_mat = np.mean(cv_confusion_mat, axis=0)
print("tot_confusion_mat: ", tot_confusion_mat)
confusion_label = ["tn", "fp", "fn", "tp"]
# Extracting for readability
tn = tot_confusion_mat[0]
fp = tot_confusion_mat[1]
fn = tot_confusion_mat[2]
tp = tot_confusion_mat[3]
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

# Showing results
for i in range(0,len(tot_confusion_mat)):
    print(confusion_label[i], ': ', tot_confusion_mat[i])
print("###### ---------Overall Results --------- ######")
print("accuracy: ", tot_accuracy)
print("auroc: ", tot_auroc)
print("confusion_mat: ", tot_confusion_mat)
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)

### Saving Metrics in the log file
metric_dict = {
    'True Negative': tn,
    'True Positive': tp,
    'False Negative': fn,
    'False Positive': fp,
    'AUROC': tot_auroc,
    'Accuracy': tot_accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1': f1
}
with open("../result/master_log.txt", "a") as file:
    print(metric_dict, file=file)

# results = cross_validate(classifier, train_X, train_y, cv=kfold, verbose=2,
#                          fit_params={'callbacks': [csv_logger, checkpointer, tensorboard_callback],
#                                      'class_weight': class_weights
#                                      },
#                          return_train_score=True, return_estimator=True)

# print("Baseline -  mean: ", results['test_score'].mean(), " std: ", results['test_score'].std())
# print("printing results: ", results)
# model_tuple = results['estimator']
# print("model_tuple: ", model_tuple)
# model = results['estimator'][len(model_tuple)-1]
# print("model: ", model)


### ---------------- Saving the model ---------------- ###

# ### Saving Entire Model or Loading 
# # serialize model to JSON
# model_json = model.to_json()
# # model_name = "model_"+ datetime.now().strftime("%Y%m%d-%H%M") + ".json"
# model_name = "model_" + "build1" +".json"
# with open(model_name, "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# # weights_name = "model_"+datetime.now().strftime("%Y%m%d-%H%M")+".h5"
# weights_name = "model_" + "1" +".h5"
# model.model.save_weights(weights_name)
# print("Saved model to disk")

# # ### Loading weights
# # # load json and create model
# # json_file = open('model_1.json', 'r')
# # loaded_model_json = json_file.read()
# # json_file.close()
# # model = model_from_json(loaded_model_json)
# # # load weights into new model
# # model.load_weights("model_1.h5")
# # model.compile(optimizer= HP['OPTIMIZER'], loss= HP['LOSS'], metrics=[HP['METRICS']])
# # print("Loaded model from disk")


# ### ---------------- Evaluating the model ---------------- ###

# # Testing the model
# # evaluation = model.evaluate(test_X, test_y, verbose = 2)
# # full_predictions = model.predict_classes(test_X)
# evaluation = model.score(test_X, test_y, verbose = 2)
# full_predictions = model.predict(test_X)
# # y_score = np.ravel(model.predict_proba(test_X))
# y_score = model.predict_proba(test_X)
# print("length of test_y: ", len(test_y))
# print("length of score_y: ", len(y_score))

# ### ---------------- Examining metrics ---------------- ###

# tn, fp, fn, tp = confusion_matrix(test_y, full_predictions).ravel()
# confusion_metrics = [tn, fp, fn, tp]
# confusion_label = ["tn", "fp", "fn", "tp"]
# for i in range(0,len(confusion_metrics)):
#     print(confusion_label[i], ': ', confusion_metrics[i])

# print("y_score: ", y_score)
# # print("y_score[:,0]: ", y_score[:,1])
# # fper, tper, thresholds = roc_curve(test_y, y_score)
# fper, tper, thresholds = roc_curve(test_y, y_score[:,1])
# print("length of fper: " , len(fper))
# print("length of tper: " , len(tper))
# print("contents of fper: " , fper)
# print("contents of tper: " , tper)
# auroc = roc_auc_score(test_y, full_predictions)
# print("auroc: ", auroc)
# plot_roc_curve(fper, tper, HP['NAME'])
# # small_predictions = model.predict(test_X[94:104])
# # print(np.argmax(small_predictions, axis=1)) # [7, 2, 1, 0, 4]



# ### ---------------- Plotting the data ---------------- ###

# # Generating a confusion matrix
# skplt.metrics.plot_confusion_matrix(test_y, full_predictions, normalize=True)
# confusion_mat_string = "../result/confusion_mat_" + HP['NAME']+ datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
# plt.savefig(confusion_mat_string)
# plt.show()

# # Generating precision-recall curve
# model_probas=model.predict_proba(test_X, batch_size=HP['BATCH_SIZE'])
# skplt.metrics.plot_precision_recall(test_y, model_probas)
# precision_recall_curve_string = "../result/precision_recall_curve_" + HP['NAME'] + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png"
# plt.savefig(precision_recall_curve_string)
# # train_sizes, train_scores, valid_scores = learning_curve(model, train_X, train_y,
# #                                                          train_sizes=[50, 80, 110],
# #                                                          cv=HP['EPOCHS'])
# plt.show()

# # results['test_score'].mean()
# ## Plotting Validation
# # Create range of values for parameter
# # Calculate mean and standard deviation for training set scores
# # train_mean = np.mean(train_scores, axis=1)
# # train_std = np.std(train_scores, axis=1)
# # # train_mean = results['train_score'].mean()
# # # train_std = results['train_score'].std()
# # # Calculate mean and standard deviation for test set scores
# # # test_mean = results['test_score'].mean()
# # # test_std = results['test_score'].std()
# # test_mean = np.mean(valid_scores, axis=1)
# # test_std = np.std(valid_scores, axis=1)

# # # Plot mean accuracy scores for training and test sets
# # plt.plot(train_sizes, train_mean, label="Training score", color="black")
# # plt.plot(train_sizes, test_mean, label="Cross-validation score", color="dimgrey")

# # # Plot accurancy bands for training and test sets
# # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="gray")
# # plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="gainsboro")

# # # Create plot
# # plt.title("Validation Curve")
# # plt.xlabel("Number Of Trees")
# # plt.ylabel("Accuracy Score")
# # plt.tight_layout()
# # plt.legend(loc="best")
# # plt.show()

# # # plt.plot(model.history['acc'])
# # # plt.plot(model.history['val_acc'])
# # # plt.title('model accuracy')
# # # plt.ylabel('accuracy')
# # # plt.xlabel('epoch')
# # # plt.legend(['train', 'val'], loc='upper left')
# # # plt.show()
