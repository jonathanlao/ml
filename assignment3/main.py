import numpy as np
import os.path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from scipy.stats import kurtosis
import time
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
ROOT = os.path.abspath(os.path.dirname(__file__))
FIGURE = 0


def plot(x, y, x_axis, y_axis, title, file_name):
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.xticks(range(len(x)))
    plt.savefig('output/'+ file_name + '.png')
    plt.close()


def load_data(dataset, datatype):
    x_path = os.path.join(ROOT, "../data/"+ dataset + "_"+ datatype +"_features.csv")
    features = np.genfromtxt(x_path, delimiter=',')

    y_path = x_path = os.path.join(ROOT, "../data/"+ dataset + "_"+datatype+"_labels.csv")
    labels = np.genfromtxt(y_path, delimiter=',')

    return features, labels

# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
# function returns WSS score for k values from 1 to kmax
def kmeans_elbow(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse


def gmm_model(points, kmax):
    bics = []
    scores = []
    for k in range(1, kmax+1):
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit(points)
        bics.append(gmm.bic(points))

        if k > 1:
            labels = gmm.predict(points)
            scores.append(silhouette_score(points, labels, metric = 'euclidean'))
        
    return bics, scores


def kmeans_silhouette(points, kmax):
    scores = []
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        labels = kmeans.labels_
        scores.append(silhouette_score(points, labels, metric = 'euclidean'))
    
    return scores


def experiment(x_train, y_train, x_test, y_test, clf, name):
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()
    fit_time = end-start
    print(name,' Fit Time = ', end-start)
    start = time.time()
    pred = clf.predict(x_test)
    end = time.time()
    # print(name, ' Predict Time = ', end-start)
    final_score = clf.score(x_test, y_test)
    print(name, ' Score', final_score)
    return (fit_time, final_score)
    
    # matrix = confusion_matrix(y_test, pred)

def step_four(x_train, y_train):
    x_test, y_test = load_data('dataset1', 'test')
    clf = MLPClassifier((110,110), max_iter=150, random_state = RANDOM_STATE)
    experiment(x_train, y_train, x_test, y_test, clf, 'NN')

    for i in range(2,9):
        print('Number of Components:', i)
        kpca = KernelPCA(kernel='rbf', n_components=i, random_state=RANDOM_STATE)
        start = time.time()
        kpca.fit(x_train)
        end = time.time()
        kpca_features = kpca.transform(x_train)
        print('KPCA Fit Time = ', end-start)

        kpca_test = KernelPCA(kernel='rbf', n_components=i, random_state=RANDOM_STATE)
        kpca_test.fit(x_test)
        kpca_features_test = kpca_test.transform(x_test)
        experiment(kpca_features, y_train, kpca_features_test, y_test, clf, 'KPCA #'+str(i))

    for i in range(2,9):
        print('Number of Components:', i)
        times = []
        scores =[]
        for j in range(10):
            rca = GaussianRandomProjection(n_components=i)
            start = time.time()
            rca.fit(x_train)
            end = time.time()
            rca_features = rca.transform(x_train)

            rca.fit(x_test)
            rca_features_test = rca.transform(x_test)
            fit_time, score = experiment(rca_features, y_train, rca_features_test, y_test, clf, 'RCA #'+str(i))
            times.append(end-start)
            scores.append(score)
        print('RCA Fit Time Average', np.mean(times))
        print('RCA Score Average', np.mean(scores))


    for i in range(2,9):
        print('Number of Components:', i)
        ica = FastICA(n_components=i, random_state=RANDOM_STATE)
        start = time.time()
        ica.fit(x_train)
        end = time.time()
        ica_features = ica.transform(x_train)

        ica_test = FastICA(n_components=i, random_state = RANDOM_STATE)
        ica_test.fit(x_test)
        print('ICA Fit Time = ', end-start)

        ica_features_test = ica_test.transform(x_test)
        experiment(ica_features, y_train, ica_features_test, y_test, clf, 'ICA #'+str(i))


    for i in range(2,9):
        print('Number of Components:', i)
        pca = PCA(n_components=i)
        start = time.time()
        pca.fit(x_train)
        end = time.time()
        pca_features = pca.transform(x_train)
        print('PCA Fit Time = ', end-start)

        pca_test = PCA(n_components=i, random_state = RANDOM_STATE)
        pca_test.fit(x_test)
        pca_features_test = pca_test.transform(x_test)
        experiment(pca_features, y_train, pca_features_test, y_test, clf, 'PCA #'+str(i))



def step_five(x_train, y_train):
    x_test, y_test = load_data('dataset1', 'test')
    clf = MLPClassifier((110,110), max_iter=150, random_state = RANDOM_STATE)
    
    kmeans = KMeans(n_clusters = 2).fit(x_train)
    pred_clusters = kmeans.predict(x_train)
    new_features = np.hstack((x_train, np.array([pred_clusters]).T))

    test_clusters = kmeans.predict(x_test)
    test_features = np.hstack((x_test, np.array([test_clusters]).T))

    experiment(new_features, y_train, test_features, y_test, clf, 'KMeans')

    gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(x_train)
    pred_clusters = gmm.predict(x_train)
    new_features = np.hstack((x_train, np.array([pred_clusters]).T))

    test_clusters = gmm.predict(x_test)
    test_features = np.hstack((x_test, np.array([test_clusters]).T))
    experiment(new_features, y_train, test_features, y_test, clf, 'GMM')




if __name__ == "__main__":    
    features1, labels1 = load_data('dataset1', 'train')
    features2, labels2 = load_data('dataset2', 'train')

    kmax = 20

    # Step 0
    df = pd.DataFrame(features1)
    df["y"] = labels1
    sns.pairplot(df, vars=df.columns[:-1], hue="y")
    plt.savefig('output/alpha_pairwise.png')
    plt.close()

    df2 = pd.DataFrame(features2)
    df2["y"] = labels2
    sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    plt.savefig('output/beta_pairwise.png')
    plt.close()

    # Step 1
    kmax = 20
    xaxis = list(range(1,kmax+1))
    elbows1 = kmeans_elbow(features1, kmax)
    elbows2 = kmeans_elbow(features2, kmax)
    plot(xaxis, elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - K-Means', 'alpha_elbow')
    plot(xaxis, elbows2, 'k', 'w/in cluster sum of squared error', 'Beta - K-Means', 'beta_elbow')

    bics1, sils1 = gmm_model(features1, kmax)
    bics2, sils2 = gmm_model(features2, kmax)
    plot(xaxis, bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_bic')
    plot(xaxis, bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_bic')

    xaxis = list(range(2,kmax+1))
    plot(xaxis, sils1, 'num clusters', 'silhouette score', 'Alpha - GMM', 'alpha_gmm_silhouette')
    plot(xaxis, sils2, 'num clusters', 'silhouette score', 'Beta - GMM', 'beta_gmm_silhouette')
    
    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(features1)
    pred_clusters = gmm.predict(features1)
    print(homogeneity_score(labels1, pred_clusters))
    gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(features1)
    pred_clusters = gmm.predict(features1)
    print(homogeneity_score(labels1, pred_clusters))

    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(features2)
    pred_clusters = gmm.predict(features2)
    print(homogeneity_score(labels2, pred_clusters))
    gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(features2)
    pred_clusters = gmm.predict(features2)
    print(homogeneity_score(labels2, pred_clusters))

    xaxis = list(range(2,kmax+1))
    sils1 = kmeans_silhouette(features1, kmax)
    sils2 = kmeans_silhouette(features2, kmax)
    plot(xaxis, sils1, 'k', 'silhouette score', 'Alpha - K-Means', 'alpha_silhouette')
    plot(xaxis, sils2, 'k', 'silhouette score', 'Beta - K-Means', 'beta_silhouette')

    kmeans = KMeans(n_clusters = 2).fit(features1)
    pred_clusters = kmeans.predict(features1)
    print(homogeneity_score(labels1, pred_clusters))
    kmeans = KMeans(n_clusters = 9).fit(features1)
    pred_clusters = kmeans.predict(features1)
    print(homogeneity_score(labels1, pred_clusters))

    kmeans = KMeans(n_clusters = 2).fit(features2)
    pred_clusters = kmeans.predict(features2)
    print(homogeneity_score(labels2, pred_clusters))
    kmeans = KMeans(n_clusters = 4).fit(features2)
    pred_clusters = kmeans.predict(features2)
    print(homogeneity_score(labels2, pred_clusters))

    # Step 2&3 PCA
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(features1)
    print(np.around(pca.explained_variance_, decimals=7, out=None))

    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(features2)
    print(np.around(pca.explained_variance_, decimals=7, out=None))
    
    pca = PCA(n_components=6, random_state=RANDOM_STATE)
    pca.fit(features1)
    pca_features1 = pca.transform(features1)
    df1 = pd.DataFrame(pca_features1)
    df1["y"] = labels1
    sns.pairplot(df1, vars=df1.columns[:-1], hue="y")
    plt.savefig('output/alpha_pca_pairwise.png')
    plt.close()

    pca = PCA(n_components=4, random_state=RANDOM_STATE)
    pca.fit(features2)
    pca_features2 = pca.transform(features2)
    df2 = pd.DataFrame(pca_features2)
    df2["y"] = labels2
    sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    plt.savefig('output/beta_pca_pairwise.png')
    plt.close()


    #K-Means PCA
    elbows1 = kmeans_elbow(pca_features1, kmax)
    plot(list(range(1,kmax+1)), elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - PCA K-Means', 'alpha_pca_elbow')
    kmeans = KMeans(n_clusters = 5, random_state=RANDOM_STATE).fit(pca_features1)
    pred_clusters = kmeans.predict(pca_features1)
    print(homogeneity_score(labels1, pred_clusters))

    elbows2 = kmeans_elbow(pca_features2, kmax)
    plot(list(range(1,kmax+1)), elbows2, 'k', 'w/in cluster sum of squared error', 'Alpha - PCA K-Means', 'beta_pca_elbow')
    kmeans = KMeans(n_clusters = 5, random_state=RANDOM_STATE).fit(pca_features2)
    pred_clusters = kmeans.predict(pca_features2)
    print(homogeneity_score(labels2, pred_clusters))

    # GMM PCA
    bics1, sils1 = gmm_model(pca_features1, kmax)
    bics2, sils2 = gmm_model(pca_features2, kmax)
    plot(list(range(1,kmax+1)), bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_pca_bic')
    plot(list(range(1,kmax+1)), bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_pca_bic')

    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(pca_features1)
    pred_clusters = gmm.predict(pca_features1)
    print(homogeneity_score(labels1, pred_clusters))
    gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(pca_features2)
    pred_clusters = gmm.predict(pca_features2)
    print(homogeneity_score(labels2, pred_clusters))

 

    # Step 2&3 ICA
    kurts = []
    for i in range(2, 18):
        ica = FastICA(n_components=i, random_state=RANDOM_STATE)
        ica = ica.fit(features1)
        ica_features = ica.transform(features1)
        # print(ica_features.shape)
        tmp = pd.DataFrame(ica_features)
        avg_kurtosis = np.mean(np.array(tmp.kurt(axis=0)))
        avg_kurtosis = np.around(avg_kurtosis, decimals=4, out=None)
        kurts.append(avg_kurtosis)
    print(kurts)

    kurts2 = []
    for i in range(2, 8):
        ica = FastICA(n_components=i, random_state=RANDOM_STATE)
        ica = ica.fit(features2)
        ica_features = ica.transform(features2)
        # print(ica_features.shape)
        tmp = pd.DataFrame(ica_features)
        avg_kurtosis = np.mean(np.array(tmp.kurt(axis=0)))
        avg_kurtosis = np.around(avg_kurtosis, decimals=4, out=None)
        kurts2.append(avg_kurtosis)
    print(kurts2)
        

    ica = FastICA(random_state=RANDOM_STATE, n_components=6)
    ica.fit(features1)
    ica_features1 = ica.transform(features1)
    elbows1 = kmeans_elbow(ica_features1, kmax)
    plot(list(range(1,kmax+1)), elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - ICA K-Means', 'alpha_ica_elbow')
    kmeans = KMeans(n_clusters = 6).fit(ica_features1)
    pred_clusters = kmeans.predict(ica_features1)
    print(homogeneity_score(labels1, pred_clusters))

    df1 = pd.DataFrame(ica_features1)
    df1["y"] = labels1
    sns.pairplot(df1, vars=df1.columns[:-1], hue="y")
    plt.savefig('output/alpha_ica_pairwise.png')
    plt.close()

    ica = FastICA(random_state=RANDOM_STATE, n_components=4)
    ica.fit(features2)
    ica_features2 = ica.transform(features2)
    elbows2 = kmeans_elbow(ica_features2, kmax)
    plot(list(range(1,kmax+1)), elbows2, 'k', 'w/in cluster sum of squared error', 'Beta - ICA K-Means', 'beta_ica_elbow')
    kmeans = KMeans(n_clusters = 4).fit(ica_features1)
    pred_clusters = kmeans.predict(ica_features1)
    print(homogeneity_score(labels1, pred_clusters))

    df2 = pd.DataFrame(ica_features2)
    df2["y"] = labels2
    sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    plt.savefig('output/beta_ica_pairwise.png')
    plt.close()

    # GMM ICA
    bics1, sils1 = gmm_model(ica_features1, kmax)
    bics2, sils2 = gmm_model(ica_features2, kmax)
    plot(list(range(1,kmax+1)), bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_ica_bic')
    plot(list(range(1,kmax+1)), bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_ica_bic')

    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(ica_features1)
    pred_clusters = gmm.predict(ica_features1)
    print(homogeneity_score(labels1, pred_clusters))
    gmm = GaussianMixture(n_components=5, random_state=RANDOM_STATE).fit(ica_features2)
    pred_clusters = gmm.predict(ica_features2)
    print(homogeneity_score(labels2, pred_clusters))


    # Step 2&3 RCA
    errs=[]
    for i in range(1,19):
        err=[]
        for j in range(10):
            rca = GaussianRandomProjection(n_components=i, random_state=RANDOM_STATE)
            rca.fit(features1)
            rca_features = rca.transform(features1)

            inverse_data = np.linalg.pinv(rca.components_.T)
            reconstructed_data = rca_features.dot(inverse_data)

            mse = (np.square(features1 - reconstructed_data)).mean(axis=None)
            err.append(mse)
        errs.append(np.around(np.mean(err), decimals=7))
        # print('Num Components', i, np.mean(err))
    print(errs)

    errs=[]
    for i in range(1,9):
        err=[]
        for j in range(10):
            rca = GaussianRandomProjection(n_components=i, random_state=RANDOM_STATE)
            rca.fit(features2)
            rca_features = rca.transform(features2)

            inverse_data = np.linalg.pinv(rca.components_.T)
            reconstructed_data = rca_features.dot(inverse_data)

            mse = (np.square(features2 - reconstructed_data)).mean(axis=None)
            err.append(mse)
        errs.append(np.around(np.mean(err), decimals=7))
        # print('Num Components', i, np.mean(err))
    print(errs)

    rca = GaussianRandomProjection(n_components=6, random_state=RANDOM_STATE)
    rca.fit(features1)
    rca_features1 = rca.transform(features1)
    df1 = pd.DataFrame(rca_features1)
    df1["y"] = labels1
    sns.pairplot(df1, vars=df1.columns[:-1], hue="y")
    plt.savefig('output/alpha_rca_pairwise.png')
    plt.close()

    rca = GaussianRandomProjection(n_components=4, random_state=RANDOM_STATE)
    rca.fit(features2)
    rca_features2 = rca.transform(features2)
    df2 = pd.DataFrame(rca_features2)
    df2["y"] = labels2
    sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    plt.savefig('output/beta_rca_pairwise.png')
    plt.close()


    #K-Means RCA
    elbows1 = kmeans_elbow(rca_features1, kmax)
    plot(list(range(1,kmax+1)), elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - RCA K-Means', 'alpha_rca_elbow')
    kmeans = KMeans(n_clusters = 4).fit(rca_features1)
    pred_clusters = kmeans.predict(rca_features1)
    print(homogeneity_score(labels1, pred_clusters))

    elbows2 = kmeans_elbow(rca_features2, kmax)
    plot(list(range(1,kmax+1)), elbows2, 'k', 'w/in cluster sum of squared error', 'Alpha - RCA K-Means', 'beta_rca_elbow')
    kmeans = KMeans(n_clusters = 4).fit(rca_features2)
    pred_clusters = kmeans.predict(rca_features2)
    print(homogeneity_score(labels2, pred_clusters))

    # GMM RCA
    bics1, sils1 = gmm_model(rca_features1, kmax)
    bics2, sils2 = gmm_model(rca_features2, kmax)
    plot(list(range(1,kmax+1)), bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_rca_bic')
    plot(list(range(1,kmax+1)), bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_rca_bic')

    gmm = GaussianMixture(n_components=3, random_state=RANDOM_STATE).fit(rca_features1)
    pred_clusters = gmm.predict(rca_features1)
    print(homogeneity_score(labels1, pred_clusters))
    gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(rca_features2)
    pred_clusters = gmm.predict(rca_features2)
    print(homogeneity_score(labels2, pred_clusters))


    # Step 2&3 KPCA
    kpca = KernelPCA(kernel='rbf', n_components=18, random_state=RANDOM_STATE)
    kpca.fit(features1)
    print(np.around(kpca.lambdas_, decimals=7, out=None))

    kpca = KernelPCA(kernel='rbf', n_components=8, random_state=RANDOM_STATE)
    kpca.fit(features2)
    print(np.around(kpca.lambdas_, decimals=7, out=None))
    
    kpca = KernelPCA(kernel='rbf', n_components=6, random_state=RANDOM_STATE)
    kpca.fit(features1)
    kpca_features1 = kpca.transform(features1)
    df1 = pd.DataFrame(kpca_features1)
    df1["y"] = labels1
    sns.pairplot(df1, vars=df1.columns[:-1], hue="y")
    plt.savefig('output/alpha_kpca_pairwise.png')
    plt.close()

    kpca = KernelPCA(kernel='rbf', n_components=4, random_state=RANDOM_STATE)
    kpca.fit(features2)
    kpca_features2 = kpca.transform(features2)
    df2 = pd.DataFrame(kpca_features2)
    df2["y"] = labels2
    sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    plt.savefig('output/beta_kpca_pairwise.png')
    plt.close()


    #K-Means PCA
    elbows1 = kmeans_elbow(kpca_features1, kmax)
    plot(list(range(1,kmax+1)), elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - KPCA K-Means', 'alpha_kpca_elbow')
    kmeans = KMeans(n_clusters = 5,  random_state=RANDOM_STATE).fit(kpca_features1)
    pred_clusters = kmeans.predict(kpca_features1)
    print(homogeneity_score(labels1, pred_clusters))

    elbows2 = kmeans_elbow(kpca_features2, kmax)
    plot(list(range(1,kmax+1)), elbows2, 'k', 'w/in cluster sum of squared error', 'Alpha - KPCA K-Means', 'beta_kpca_elbow')
    kmeans = KMeans(n_clusters = 6,  random_state=RANDOM_STATE).fit(kpca_features2)
    pred_clusters = kmeans.predict(kpca_features2)
    print(homogeneity_score(labels2, pred_clusters))

    # GMM PCA
    bics1, sils1 = gmm_model(kpca_features1, kmax)
    bics2, sils2 = gmm_model(kpca_features2, kmax)
    plot(list(range(1,kmax+1)), bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_kpca_bic')
    plot(list(range(1,kmax+1)), bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_kpca_bic')

    gmm = GaussianMixture(n_components=7, random_state=RANDOM_STATE).fit(kpca_features1)
    pred_clusters = gmm.predict(kpca_features1)
    print(homogeneity_score(labels1, pred_clusters))
    gmm = GaussianMixture(n_components=8, random_state=RANDOM_STATE).fit(kpca_features2)
    pred_clusters = gmm.predict(kpca_features2)
    print(homogeneity_score(labels2, pred_clusters))

    step_four(features1, labels1)

    step_five(features1, labels1)