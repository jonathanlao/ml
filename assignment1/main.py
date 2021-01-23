import numpy as np
import os.path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

RANDOM_STATE = 42
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


def classify(x_train, x_test, y_train, y_test, dataset, clf, clf_name):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    final_score = clf.score(x_test, y_test)
    matrix = confusion_matrix(y_test, pred)
    
    print(y_test)
    print(final_score)
    print(matrix)
    if dataset == 'wine':
            disp = plot_confusion_matrix(clf, x_test, y_test,
                                            display_labels=['Cultivar 1', 'Cultivar 2', 'Cultivar 3'],
                                            cmap=plt.cm.Blues,
                                            # values_format=values_format,
                                            normalize=None)
            plt.title('Wine Confusion Matrix')
            plt.savefig(clf_name+'_Wine.png')

    elif dataset == 'diabetes':
        disp = plot_confusion_matrix(clf, x_test, y_test,
                                        display_labels=['No', 'Yes'],
                                        cmap=plt.cm.Blues,
                                        # values_format=values_format,
                                        normalize=None)
        plt.title('Diabetes Confusion Matrix')
        plt.savefig(clf_name+'_Diabetes.png')


def decision_tree(x_train, x_test, y_train, y_test, dataset):
    dt = DecisionTreeClassifier()
    param_grid = {'max_depth': np.arange(1, 10)}
    gscv = GridSearchCV(dt, param_grid, cv=5)
    gscv.fit(x_train, y_train)
    depth = gscv.best_params_['max_depth']
    # print(depth)

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=depth)
    classify(x_train, x_test, y_train, y_test, dataset, dt, 'DT')


def neural_network(x_train, x_test, y_train, y_test, dataset):
    print('hello!')


def svm(x_train, x_test, y_train, y_test, dataset):
    svm = LinearSVC(random_state=RANDOM_STATE)
    classify(x_train, x_test, y_train, y_test, dataset, svm, 'SVM')


def knn(x_train, x_test, y_train, y_test, dataset):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 10)}
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)
    knn_gscv.fit(x_train, y_train)
    k = knn_gscv.best_params_['n_neighbors']
    # print(k)

    knn = KNeighborsClassifier(n_neighbors=k)
    classify(x_train, x_test, y_train, y_test, dataset, knn, 'KNN')


def run_experiments(features, labels, dataset):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_STATE)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # decision_tree(x_train, x_test, y_train, y_test, dataset)
    svm(x_train, x_test, y_train, y_test, dataset)
    # knn(x_train, x_test, y_train, y_test, dataset)
    
    
if __name__ == "__main__":
    wine_features, wine_labels = load_data('wine', False)
    diabetes_features, diabetes_labels = load_data('diabetes', True)
    run_experiments(wine_features, wine_labels, 'wine')
    run_experiments(diabetes_features, diabetes_labels, 'diabetes')

