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


def gmm(points, kmax):
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


if __name__ == "__main__":    
    features1, labels1 = load_data('dataset1', 'train')
    features2, labels2 = load_data('dataset2', 'train')

    kmax = 20





    # Step 2&3 PCA
    pca = PCA()
    pca.fit(features1)
    print(np.around(pca.explained_variance_, decimals=7, out=None))

    pca = PCA()
    pca.fit(features2)
    print(np.around(pca.explained_variance_, decimals=7, out=None))
    
    pca = PCA(n_components=6)
    pca.fit(features1)
    pca_features1 = pca.transform(features1)
    df1 = pd.DataFrame(pca_features1)
    df1["y"] = labels1
    sns.pairplot(df1, vars=df1.columns[:-1], hue="y")
    plt.savefig('output/alpha_pca_pairwise.png')
    plt.close()

    pca = PCA(n_components=4)
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
    kmeans = KMeans(n_clusters = 5).fit(pca_features1)
    pred_clusters = kmeans.predict(pca_features1)
    print(homogeneity_score(labels1, pred_clusters))

    elbows2 = kmeans_elbow(pca_features2, kmax)
    plot(list(range(1,kmax+1)), elbows2, 'k', 'w/in cluster sum of squared error', 'Alpha - PCA K-Means', 'beta_pca_elbow')
    kmeans = KMeans(n_clusters = 5).fit(pca_features2)
    pred_clusters = kmeans.predict(pca_features2)
    print(homogeneity_score(labels2, pred_clusters))

    # GMM PCA
    bics1, sils1 = gmm(pca_features1, kmax)
    bics2, sils2 = gmm(pca_features2, kmax)
    plot(list(range(1,kmax+1)), bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_pca_bic')
    plot(list(range(1,kmax+1)), bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_pca_bic')

    gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(pca_features1)
    pred_clusters = gmm.predict(pca_features1)
    print(homogeneity_score(labels1, pred_clusters))
    gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(pca_features2)
    pred_clusters = gmm.predict(pca_features2)
    print(homogeneity_score(labels2, pred_clusters))

 

    ## Step 2&3 ICA
    # kurts = []
    # for i in range(2, 18):
    #     ica = FastICA(n_components=i, random_state=RANDOM_STATE)
    #     ica = ica.fit(features1)
    #     ica_features = ica.transform(features1)
    #     # print(ica_features.shape)
    #     tmp = pd.DataFrame(ica_features)
    #     avg_kurtosis = np.mean(np.array(tmp.kurt(axis=0)))
    #     avg_kurtosis = np.around(avg_kurtosis, decimals=4, out=None)
    #     kurts.append(avg_kurtosis)
    # print(kurts)

    # kurts2 = []
    # for i in range(2, 8):
    #     ica = FastICA(n_components=i, random_state=RANDOM_STATE)
    #     ica = ica.fit(features2)
    #     ica_features = ica.transform(features2)
    #     # print(ica_features.shape)
    #     tmp = pd.DataFrame(ica_features)
    #     avg_kurtosis = np.mean(np.array(tmp.kurt(axis=0)))
    #     avg_kurtosis = np.around(avg_kurtosis, decimals=4, out=None)
    #     kurts2.append(avg_kurtosis)
    # print(kurts2)
        

    # ica = FastICA(random_state=RANDOM_STATE, n_components=6)
    # ica.fit(features1)
    # ica_features1 = ica.transform(features1)
    # elbows1 = kmeans_elbow(ica_features1, kmax)
    # plot(list(range(1,kmax+1)), elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - ICA K-Means', 'alpha_ica_elbow')
    # kmeans = KMeans(n_clusters = 6).fit(ica_features1)
    # pred_clusters = kmeans.predict(ica_features1)
    # print(homogeneity_score(labels1, pred_clusters))

    # df1 = pd.DataFrame(ica_features1)
    # df1["y"] = labels1
    # sns.pairplot(df1, vars=df1.columns[:-1], hue="y")
    # plt.savefig('output/alpha_ica_pairwise.png')
    # plt.close()

    # ica = FastICA(random_state=RANDOM_STATE, n_components=4)
    # ica.fit(features2)
    # ica_features2 = ica.transform(features2)
    # elbows2 = kmeans_elbow(ica_features2, kmax)
    # plot(list(range(1,kmax+1)), elbows2, 'k', 'w/in cluster sum of squared error', 'Beta - ICA K-Means', 'beta_ica_elbow')
    # kmeans = KMeans(n_clusters = 4).fit(ica_features1)
    # pred_clusters = kmeans.predict(ica_features1)
    # print(homogeneity_score(labels1, pred_clusters))

    # df2 = pd.DataFrame(ica_features2)
    # df2["y"] = labels2
    # sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    # plt.savefig('output/beta_ica_pairwise.png')
    # plt.close()

    # # GMM ICA
    # bics1, sils1 = gmm(ica_features1, kmax)
    # bics2, sils2 = gmm(ica_features2, kmax)
    # plot(list(range(1,kmax+1)), bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_ica_bic')
    # plot(list(range(1,kmax+1)), bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_ica_bic')

    # gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(ica_features1)
    # pred_clusters = gmm.predict(ica_features1)
    # print(homogeneity_score(labels1, pred_clusters))
    # gmm = GaussianMixture(n_components=5, random_state=RANDOM_STATE).fit(ica_features2)
    # pred_clusters = gmm.predict(ica_features2)
    # print(homogeneity_score(labels2, pred_clusters))




    ## Step 0
    # df = pd.DataFrame(features1)
    # df["y"] = labels1
    # sns.pairplot(df, vars=df.columns[:-1], hue="y")
    # plt.savefig('output/alpha_pairwise.png')
    # plt.close()

    # df2 = pd.DataFrame(features2)
    # df2["y"] = labels2
    # sns.pairplot(df2, vars=df2.columns[:-1], hue="y")
    # plt.savefig('output/beta_pairwise.png')
    # plt.close()

    ## Step 1
    # kmax = 20
    # xaxis = list(range(1,kmax+1))
    # elbows1 = kmeans_elbow(features1, kmax)
    # elbows2 = kmeans_elbow(features2, kmax)
    # plot(xaxis, elbows1, 'k', 'w/in cluster sum of squared error', 'Alpha - K-Means', 'alpha_elbow')
    # plot(xaxis, elbows2, 'k', 'w/in cluster sum of squared error', 'Beta - K-Means', 'beta_elbow')

    # bics1, sils1 = gmm(features1, kmax)
    # bics2, sils2 = gmm(features2, kmax)
    # plot(xaxis, bics1, 'num clusters', 'bayesian information criterion', 'Alpha - GMM', 'alpha_bic')
    # plot(xaxis, bics2, 'num clusters', 'bayesian information criterion', 'Beta - GMM', 'beta_bic')

    # xaxis = list(range(2,kmax+1))
    # plot(xaxis, sils1, 'num clusters', 'silhouette score', 'Alpha - GMM', 'alpha_gmm_silhouette')
    # plot(xaxis, sils2, 'num clusters', 'silhouette score', 'Beta - GMM', 'beta_gmm_silhouette')
    
    # gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(features1)
    # pred_clusters = gmm.predict(features1)
    # print(homogeneity_score(labels1, pred_clusters))
    # gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(features1)
    # pred_clusters = gmm.predict(features1)
    # print(homogeneity_score(labels1, pred_clusters))

    # gmm = GaussianMixture(n_components=2, random_state=RANDOM_STATE).fit(features2)
    # pred_clusters = gmm.predict(features2)
    # print(homogeneity_score(labels2, pred_clusters))
    # gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE).fit(features2)
    # pred_clusters = gmm.predict(features2)
    # print(homogeneity_score(labels2, pred_clusters))

    # xaxis = list(range(2,kmax+1))
    # sils1 = kmeans_silhouette(features1, kmax)
    # sils2 = kmeans_silhouette(features2, kmax)
    # plot(xaxis, sils1, 'k', 'silhouette score', 'Alpha - K-Means', 'alpha_silhouette')
    # plot(xaxis, sils2, 'k', 'silhouette score', 'Beta - K-Means', 'beta_silhouette')

    # kmeans = KMeans(n_clusters = 2).fit(features1)
    # pred_clusters = kmeans.predict(features1)
    # print(homogeneity_score(labels1, pred_clusters))
    # kmeans = KMeans(n_clusters = 9).fit(features1)
    # pred_clusters = kmeans.predict(features1)
    # print(homogeneity_score(labels1, pred_clusters))

    # kmeans = KMeans(n_clusters = 2).fit(features2)
    # pred_clusters = kmeans.predict(features2)
    # print(homogeneity_score(labels2, pred_clusters))
    # kmeans = KMeans(n_clusters = 4).fit(features2)
    # pred_clusters = kmeans.predict(features2)
    # print(homogeneity_score(labels2, pred_clusters))
