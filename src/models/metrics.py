from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import japanize_matplotlib


def cal_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall =recall_score(y_true, y_pred)
    f1 = f1_score(y_true,y_pred)
    return {'acc': acc, 'f1':f1, 'precision':precision, 'recall':recall}

def plot_cm(y_true, y_pred):
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(cmap="Blues")
    plt.xlabel('予測')
    plt.ylabel('実際')
    plt.title('混同行列')
    # plt.show()