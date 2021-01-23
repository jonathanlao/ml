import numpy as np
import os.path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


ROOT = os.path.abspath(os.path.dirname(__file__))

def load_data(dataset, outcome_last_col):
    path = os.path.join(ROOT, "../data/"+ dataset +".csv")
    data = np.genfromtxt(path, delimiter=',')
    if outcome_last_col:
        features = data[:,0:-1]
        labels = data[:,-1]
    else:
        features = data[:,1:]
        labels = data[:,0]

    return features, labels

def run_experiments():
    features, labels = load_data('wine', False)
    diabetes_features, diabetes_labels = load_data('diabetes', True)

    random_state = 42
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_state)
    scaler = preprocessing.StandardScaler().fit(x_train)

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    


if __name__ == "__main__":
    run_experiments()

