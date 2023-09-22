import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import model_selection
from math import ceil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Loads csv files and appends pixels to X and labels to y
def preprocess_data():
    data = pd.read_csv('./dataset/fer2013.csv') #csvformat => emotion,pixels,Usage
    labels = pd.read_csv('./dataset/fer2013new.csv') #csvformat => Usage,Image name,neutral,happiness,surprise,sadness,anger,disgust,fear,contempt,unknown,NF

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y


def clean_data_and_normalize(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0

    return X, y


def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size,
                                                                      random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_model_and_weights(model_path='model.json', weights_path='model.h5'):
    # Loading JSON model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Loading weights
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('Model and weights are loaded and compiled.')

    return model

def plot_roc_curve(y_test_bin, y_pred):
    n_classes = y_test_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10,8))
    for i, color in zip(range(n_classes), ['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 'magenta']):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def generate_report(model, x_test, y_test):
    y_pred = model.predict(x_test, verbose=1)

    # Convert predictions and true values to binary matrix format for multi-label ROC
    y_pred_bin = y_pred
    y_test_bin = label_binarize(np.argmax(y_test, axis=1), classes=[0, 1, 2, 3, 4, 5, 6])

    plot_roc_curve(y_test_bin, y_pred_bin)

    y_pred_bool = np.argmax(y_pred, axis=1)
    y_test_bool = np.argmax(y_test, axis=1)
    roc_auc = roc_auc_score(y_test_bin, label_binarize(y_pred_bool, classes=[0, 1, 2, 3, 4, 5, 6]), average="macro")

    print("Classification Report")
    print(classification_report(y_test_bool, y_pred_bool))
    print("ROC AUC Score:", roc_auc)

    cm = confusion_matrix(y_test_bool, y_pred_bool)
    print("Confusion Matrix")
    print(cm)

def test_model():
    X, y = preprocess_data()
    X, y = clean_data_and_normalize(X, y)
    _, _, _, _, x_test, y_test = split_data(X, y)

    model = load_model_and_weights()
    generate_report(model, x_test, y_test)

test_model()
