import numpy as np
import os.path
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate

RANDOM_STATE = 42
ROOT = os.path.abspath(os.path.dirname(__file__))
FIGURE = 0

# def create_wdbc_dataset(dataset):
#     path = os.path.join(ROOT, "../data/"+ dataset +".csv")
#     data = np.genfromtxt(path, delimiter=',')
    
#     features = data[:,2:]
#     labels = data[:,1]

#     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=150, random_state=RANDOM_STATE)

#     np.savetxt("../data/wdbc_train_features.csv", x_train, delimiter=",", fmt="%1.5f")
#     np.savetxt("../data/wdbc_train_labels.csv", y_train, delimiter=",", fmt="%d")
#     np.savetxt("../data/wdbc_test_features.csv", x_test, delimiter=",", fmt="%1.5f")
#     np.savetxt("../data/wdbc_test_labels.csv", y_test, delimiter=",", fmt="%d")
#     return


# def create_wine_dataset(dataset):
#     path = os.path.join(ROOT, "../data/"+ dataset +".csv")
#     data = np.genfromtxt(path, delimiter=';')
    
#     features = data[1:,:-1]
#     labels = data[1:,-1]

#     # unique, counts = np.unique(labels, return_counts=True)
#     # print(dict(zip(unique, counts)))

#     labels[labels < 5] = 0
#     labels[labels == 5] = 1
#     labels[labels == 6] = 1
#     labels[labels > 5] = 2

#     # unique, counts = np.unique(labels, return_counts=True)
#     # print(dict(zip(unique, counts)))

#     # random subset of features
#     features = features[:,[0,3,4,6,7,10]]

#     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=150, random_state=RANDOM_STATE)

#     # unique, counts = np.unique(y_test, return_counts=True)
#     # print(dict(zip(unique, counts)))

#     np.savetxt("../data/wine_train_features.csv", x_train, delimiter=",", fmt="%1.5f")
#     np.savetxt("../data/wine_train_labels.csv", y_train, delimiter=",", fmt="%d")
#     np.savetxt("../data/wine_test_features.csv", x_test, delimiter=",", fmt="%1.5f")
#     np.savetxt("../data/wine_test_labels.csv", y_test, delimiter=",", fmt="%d")

    # return features, labels


def create_datasets():
    # x1, y1 = make_gaussian_quantiles(mean = None, cov = 1.0, n_samples = 1000, n_features = 15, n_classes = 2, random_state = RANDOM_STATE) 
    x1, y1 = make_classification(n_samples=1000, n_features=18, n_informative=16, n_redundant=1, n_repeated=1, n_classes=2, class_sep=0.8, shuffle=True, random_state=RANDOM_STATE)
    x2, y2 = make_classification(n_samples=3000, n_features=8, n_informative=6, n_redundant=1, n_repeated=1, n_classes=2, class_sep=0.5, shuffle=True, random_state=RANDOM_STATE)

    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=200, random_state=RANDOM_STATE)
    np.savetxt("../data/dataset1_train_features.csv", x1_train, delimiter=",", fmt="%1.8f")
    np.savetxt("../data/dataset1_train_labels.csv", y1_train, delimiter=",", fmt="%d")
    np.savetxt("../data/dataset1_test_features.csv", x1_test, delimiter=",", fmt="%1.8f")
    np.savetxt("../data/dataset1_test_labels.csv", y1_test, delimiter=",", fmt="%d")

    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=200, random_state=RANDOM_STATE)
    np.savetxt("../data/dataset2_train_features.csv", x2_train, delimiter=",", fmt="%1.8f")
    np.savetxt("../data/dataset2_train_labels.csv", y2_train, delimiter=",", fmt="%d")
    np.savetxt("../data/dataset2_test_features.csv", x2_test, delimiter=",", fmt="%1.8f")
    np.savetxt("../data/dataset2_test_labels.csv", y2_test, delimiter=",", fmt="%d")
    return


def load_data(dataset):
    x_path = os.path.join(ROOT, "../data/"+ dataset + "_train_features.csv")
    features = np.genfromtxt(x_path, delimiter=',')

    y_path = x_path = os.path.join(ROOT, "../data/"+ dataset + "_train_labels.csv")
    labels = np.genfromtxt(y_path, delimiter=',')

    return features, labels

# Taken from sklearn documentation:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.4, 1.01), cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)



    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    plt.savefig('output/'+title+'.png')
    plt.close()

    return plt

# taken from sklearn documentaiton
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

from sklearn.model_selection import validation_curve
def plot_model_complexity_curve(model, title, features, labels, x_label, param_name, param_range, optional_param_range=None):

    train_scores, test_scores = validation_curve(
        model, features, labels, param_name=param_name, param_range=param_range,
        scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print(test_scores_mean)
    print(test_scores_std)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.01)
    lw = 2

    if optional_param_range:
        param_range = optional_param_range

    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")

    plt.savefig('output/'+title)
    plt.close()


def classify(x_train, x_test, y_train, y_test, dataset, clf, clf_name):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    final_score = clf.score(x_test, y_test)
    matrix = confusion_matrix(y_test, pred)
    
    # print(y_test)
    print(final_score)
    # print(matrix)
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


def svm_alpha(features, labels):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    print('SVM Alpha')
    for k in kernels:
        svm = SVC(kernel=k, random_state = RANDOM_STATE)
        scores = cross_validate(svm, features, labels, cv=5, scoring='accuracy', return_train_score=True)

        train_scores = scores['train_score']
        train_score_mean = np.mean(train_scores)
        train_score_std = np.std(train_scores)
        print('Kernel='+k)
        print('Train Scores', train_score_mean, train_score_std)

        test_scores = scores['test_score']
        test_score_mean = np.mean(test_scores)
        test_score_std = np.std(test_scores)
        print('Test Scores', test_score_mean, test_score_std)

    svm = SVC(kernel='rbf', random_state = RANDOM_STATE)
    plot_model_complexity_curve(svm, 'Alpha - SVM Model Complexity, rbf', features, labels, 'Regularization', 'C', (range(1,10)))

    svm = SVC(kernel='rbf', C=4, random_state = RANDOM_STATE)
    plot_learning_curve(svm, 'Alpha - SVM Learning Curve', features, labels, train_sizes=(10, 20, 50, 100, 200, 300, 400, 500, 600))


def svm_beta(features, labels):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    print('SVM Beta')
    for k in kernels:
        svm = SVC(kernel=k, random_state = RANDOM_STATE)
        scores = cross_validate(svm, features, labels, cv=5, scoring='accuracy', return_train_score=True)

        train_scores = scores['train_score']
        train_score_mean = np.mean(train_scores)
        train_score_std = np.std(train_scores)
        print('Kernel='+k)
        print('Train Scores', train_score_mean, train_score_std)

        test_scores = scores['test_score']
        test_score_mean = np.mean(test_scores)
        test_score_std = np.std(test_scores)
        print('Test Scores', test_score_mean, test_score_std)

    svm = SVC(kernel='rbf', random_state = RANDOM_STATE)
    plot_model_complexity_curve(svm, 'Beta - SVM Model Complexity, rbf', features, labels, 'Regularization', 'C', (range(1,20)))

    svm = SVC(kernel='rbf', C=19, random_state = RANDOM_STATE)
    plot_learning_curve(svm, 'Beta - SVM Learning Curve', features, labels, 
        train_sizes=(30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000, 2200))


def boost_alpha(features, labels):
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
    boost = AdaBoostClassifier(dt, random_state = RANDOM_STATE)
    plot_model_complexity_curve(boost, 'Alpha - AdaBoost Model Complexity, depth=8', features, labels, 'Number of Estimators', 
        'n_estimators', (30,60,90,120,150,180, 200, 230))

    plot_learning_curve(boost, 'Alpha - AdaBoost Learning Curve', features, labels, train_sizes=(10, 20, 50, 100, 200, 300, 400, 500, 600))


def boost_beta(features, labels):
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=9)
    boost = AdaBoostClassifier(dt, random_state = RANDOM_STATE)
    plot_model_complexity_curve(boost, 'Beta - AdaBoost Model Complexity, depth=9', features, labels, 'Number of Estimators', 
        'n_estimators', (30,60,90,120,150,180, 200, 230, 260, 290))

    plot_learning_curve(boost, 'Beta - AdaBoost Learning Curve', features, labels, 
        train_sizes=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000, 2200))


def knn_alpha(features, labels):
    knn = KNeighborsClassifier()
    plot_model_complexity_curve(knn, 'Alpha - KNN Model Complexity, Default', features, labels, 'K', 'n_neighbors', (range(1,30)))

    knn = KNeighborsClassifier(n_neighbors=5)
    plot_learning_curve(knn, 'Alpha - KNN Learning Curve', features, labels, train_sizes=(10, 20, 50, 100, 200, 300, 400, 500, 600))


def knn_beta(features, labels):
    knn = KNeighborsClassifier()
    plot_model_complexity_curve(knn, 'Beta - KNN Model Complexity, Default', features, labels, 'K', 'n_neighbors', (range(1,50)))

    knn = KNeighborsClassifier(n_neighbors=25)
    plot_learning_curve(knn, 'Beta - KNN Learning Curve', features, labels, 
        train_sizes=(30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000, 2200))


def decision_tree_alpha(features, labels):
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

    plot_model_complexity_curve(dt, 'Alpha - Decision Tree Model Complexity, Default', features, labels, 
        'Max Depth', 'max_depth', (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
    plot_model_complexity_curve(dt, 'Alpha - Decision Tree Model Complexity, max_depth=8', features, labels, 
        'Min samples leaf', 'min_samples_leaf', (range(2,30)))

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8, min_samples_leaf=20)
    plot_learning_curve(dt, 'Alpha - Decision Tree Learning Curve', features, labels, train_sizes=(10, 20, 50, 100, 200, 300, 400, 500, 600))


def decision_tree_beta(features, labels):
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    plot_model_complexity_curve(dt, 'Beta - Decision Tree Model Complexity, Default', features, labels, 'Max Depth', 'max_depth', (range(1,25)))

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=9)
    plot_model_complexity_curve(dt, 'Beta - Decision Tree Model Complexity, max_depth=9', features, labels, 'Min samples leaf', 'min_samples_leaf', (range(2,20)))

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=9, min_samples_leaf=10)
    plot_learning_curve(dt, 'Beta - Decision Tree, depth=10', features, labels,
        train_sizes=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000, 2200))


def neural_network_alpha(features, labels):
    # 100 units, iterations
    nn = MLPClassifier(random_state = RANDOM_STATE)
    plot_model_complexity_curve(nn, 'Alpha - NN Model Complexity, Default', features, labels, 'Iterations', 'max_iter', (10,20,30,40,60,80,100,120,140,160,180,200))

    # 150 iterations, adjust number of layers, clearly overfitting
    nn = MLPClassifier(max_iter=150, random_state = RANDOM_STATE)
    plot_model_complexity_curve(nn, 'Alpha - NN Model Complexity, 10 unit layers', features, labels, 'Number of layers', 'hidden_layer_sizes', 
        ((10,), (10,10,), (10,10,10,), (10,10,10,10), (10,10,10,10,10)), (1,2,3,4,5))

    # 150 iterations, adjust num hidden units
    nn = MLPClassifier(max_iter=150, random_state = RANDOM_STATE)
    plot_model_complexity_curve(nn, 'Alpha - NN Model Complexity, 2 layers', features, labels, 'Hidden Units Width', 'hidden_layer_sizes', 
        ((10,10), (20,20), (30,30), (40,40), (50,50), (60,60), (70,70), (80,80), (90,90),(100,100),(110,110),(120,120), (130,130), (140,140) ),
         (10,20,30,40,50,60,70,80,90,100,110,120, 130, 140))

    # 10 units, 600 iterations, learning curve
    nn = MLPClassifier((110,110), max_iter=150, random_state = RANDOM_STATE)
    plot_learning_curve(nn, 'Alpha - NN Learning Curve', features, labels, train_sizes=(10, 20, 50, 100, 200, 300, 400, 500, 600))

    # above was bias, now try reducing variance
    # nn = MLPClassifier(max_iter=150, hidden_layer_sizes=(100,), random_state = RANDOM_STATE)
    # plot_model_complexity_curve(nn, 'Alpha - NN Model Complexity, 1-layer 100 units', features, labels, 'Reglurization', 'alpha', 
    #     (0, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007,0.00008, 0.00009, 0.0001, 0.00011))


def neural_network_beta(features, labels):
    # 10 units, iterations
    nn = MLPClassifier(random_state = RANDOM_STATE)
    plot_model_complexity_curve(nn, 'Beta - NN Model Complexity, Default', features, labels, 'Iterations', 'max_iter', (10,20,30,40,60,80,100,120,140,160,180,200))

    nn = MLPClassifier(max_iter=100, random_state = RANDOM_STATE)
    plot_model_complexity_curve(nn, 'Beta - NN Model Complexity, 1 layer', features, labels, 'Hidden Units Per Layer', 'hidden_layer_sizes', 
        ((10,), (20,), (30,), (40), (50,), (60,), (70,), (80,), (90,), (100,)), 
        (range(10,101,10)))

    nn = MLPClassifier(max_iter=100, random_state = RANDOM_STATE)
    plot_model_complexity_curve(nn, 'Beta - NN Model Complexity, 50 unit layers', features, labels, 'Number of layers', 'hidden_layer_sizes', 
        ((50,), (50,50,), (50,50,50,), (50,50,50,50), (50,50,50,50,50)), (1,2,3,4,5))

    nn = MLPClassifier((50,50), max_iter=100, random_state = RANDOM_STATE)
    plot_learning_curve(nn, 'Beta - NN Learning Curve', features, labels, 
        train_sizes=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1500, 1750, 2000, 2200))



# def run_experiments(features, labels, dataset):
#     # scaler = preprocessing.StandardScaler().fit(features)
#     # features = scaler.transform(features)

#     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_STATE)

#     decision_tree_alpha(features, labels)
#     # svm(x_train, x_test, y_train, y_test, dataset)
#     # knn(x_train, x_test, y_train, y_test, dataset)
    
    
if __name__ == "__main__":
    # create_wdbc_dataset('wdbc')
    # create_wine_dataset('wine')
    # 

    # wdbc_features, wdbc_labels = load_data('wdbc')
    # wine_features, wine_labels = load_data('wine')

    # run_experiments(wine_features, wine_labels, 'wine')
    # run_experiments(wdbc_features, wdbc_labels, 'wdbc')
    
    # create_datasets()
    features1, labels1 = load_data('dataset1')
    # decision_tree_alpha(features1, labels1)
    # knn_alpha(features1, labels1)
    # svm_alpha(features1, labels1)
    # neural_network_alpha(features1, labels1)
    # boost_alpha(features1, labels1)

    features2, labels2 = load_data('dataset2')
    # decision_tree_beta(features2, labels2)
    # knn_beta(features2, labels2)
    # svm_beta(features2, labels2)
    neural_network_beta(features2, labels2)
    # boost_beta(features2, labels2)

