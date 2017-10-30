import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import argparse

from keras.datasets import cifar10, cifar100
from keras.applications import Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet
from keras.applications.inception_v3 import preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, DBSCAN, Birch
from sklearn import metrics
from sklearn.utils import shuffle


def choose_network_model(network_model='VGG16', input_layer_shape=(197, 197, 3)):
    """
    Choose the network base model.
    Params:
    - network_model: a string which represents the network model.
    - input_layer_shape: the shape of the input layer.
    Returns:
    - the network model
    """
    network_model_dictionary = {
        'Xception': Xception(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'VGG19': VGG19(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=input_layer_shape)}
    return network_model_dictionary[network_model]


def choose_dataset(dataset='cifar10'):
    dataset_dictionary = {
        'cifar10': cifar10,
        'cifar100': cifar100}
    return dataset_dictionary[dataset]


def choose_clustering_algorithm(clustering_algorithm='KMeans', n_clusters=10):
    clustering_dictionary = {
        'KMeans': KMeans(n_clusters=n_clusters),
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters)}
    return clustering_dictionary[clustering_algorithm]


def classy_evaluation(args):
    dataset = args.dataset
    network_model = args.network_model
    compute_bottleneck_features(args)


def compute_bottleneck_features(args):
    
    model_choice = args.network_model
    do_pca = args.do_pca
    pca_components = args.pca_components
    dataset = args.dataset
    clustering_algorithm = args.cluster_algorithm
    do_clustering = args.do_clustering
    do_tsne = args.do_tsne
    do_kpca = args.do_kpca

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    y_train = np.squeeze(y_train)
    print('data loaded')

    model = choose_network_model(model_choice)
    print('Model loaded.')

    # bottleneck features
    feature_file = model_choice + '_features_train.npz'
    if os.path.exists(feature_file):
        print('Bottleneck feature detected. Loading...')
        features = np.load(feature_file)['features']
    else:
        print('Bottleneck features not detected. Pre-processing data...')
        # pre-process the train data
        big_x_train = np.array([scipy.misc.imresize(x_train[i], (197, 197, 3)) for i in range(0, len(x_train))]).astype('float32')
        input_train = preprocess_input(big_x_train)
        print('Data pre-processed. Predicting bottleneck features...')
        # extract, process, and save bottleneck features
        features = model.predict(input_train)
        features = np.squeeze(features)
        print('Bottleneck features predicted. Saving...')
        np.savez(feature_file, features=features)
        print('bottleneck features saved(train)')
    
    if do_clustering == True:
#        cluster_file = clustering_algorithm + "_labels_pred.npz"
#        if os.path.exists(cluster_file):
#            print("clustering detected loading")
#            labels_pred = np.load(cluster_file)['labels_pred']
#        else:
#            print('Labels prediction not detected. Clustering...')
        for i in range(97, 103):
            print('clustering with the following number of clusters')
            print(i)
            clustering_model = choose_clustering_algorithm(clustering_algorithm, n_clusters=i)
            labels_pred = clustering_model.fit_predict(features.reshape([features.shape[0], np.prod(features.shape[1:])]).astype('float'))
            # Clustering metrics
            y_train = y_train.reshape((1, len(y_train))).ravel()
            labels_pred = labels_pred.reshape((1, len(y_train))).ravel()
            ari = metrics.adjusted_rand_score(y_train, labels_pred)
            ami = metrics.adjusted_mutual_info_score(y_train, labels_pred)
            print("ARI")
            print(ari)
            print("AMI")
            print(ami)
#        np.savez(cluster_file, labels_pred = labels_pred)

    # TODO: FORMALIZE K-FOLD
    if do_pca == True:
        pca_file = model_choice + '_pca_features.npz'
        if os.path.exists(pca_file):
            print('loading pca')
            pca_features = np.load(pca_file)['pca_features']
            pca_explained_variance = np.load(pca_file)['pca_explained_variance']
        else:
            print('computing pca features')
            pca = PCA(n_components=100)
            kpca = KernelPCA(kernel='rbf', n_components=100)
            reshaped_features = features.reshape([features.shape[0], np.prod(features.shape[1:])]).astype('float')
            pca_auc = []
            kpca_auc = []
            for i in range(11):
                shuffle_features = shuffle(reshaped_features)
                pca_features = pca.fit_transform(shuffle_features[:10000])
                kpca_features = kpca.fit_transform(shuffle_features[:10000])
                pca_explained_variance = pca.explained_variance_ratio_
                kpca_explained_variance_ = np.var(kpca_features, axis=0)
                kpca_explained_variance = kpca_explained_variance_ / np.sum(kpca_explained_variance_)
                print('calcolata la passata')
                pca_auc.append(metrics.auc(np.arange(1,101), np.cumsum(pca_explained_variance)))
                kpca_auc.append(metrics.auc(np.arange(1,101), np.cumsum(kpca_explained_variance)))
            pca_auc_out = np.mean(pca_auc)
            print('auc pca media')
            print(pca_auc_out)
            kpca_auc_out = np.mean(kpca_auc)
            print('auc kpca media')
            print(kpca_auc_out)
            plt.plot(np.cumsum(pca_explained_variance))
            plt.plot(np.cumsum(kpca_explained_variance))
            plt.xlabel('Kernel PCA components')
            plt.ylabel('Explained variance (%)')
            plt.title('Variance explained by Kernel PCA')
            plt.savefig(model_choice + '_kpca_explained_variance.jpg')

    if do_tsne == True:
        tsne_file = model_choice + '_tsne_features.npz'
        if os.path.exists(tsne_file):
            print('load tsne features')
            tsne_features = np.load(tsne_file)['tsne_features']
        else:
            print('tsne fatures not detected (test)')
            print('calculating now...')
            tsne_features = TSNE(n_components=2).fit_transform(features.reshape([pca_features.shape[0], np.prod(features.shape[1:])]).astype('float')[:25000])
            print('saving...')
            np.savez(tsne_file, tsne_features=tsne_features)
            print('tsne feature saved')
        plt.figure(figsize=(20,20))
        plt.scatter(tsne_features[:,0], tsne_features[:,1], c=plt.cm.jet(y_train/10), s=10, edgecolors='none')
        plt.title('t-SNE clusters')
        plt.savefig(model_choice + '_tsne.jpg')


    # Clustering metrics
    # y_train = y_train.reshape((1, len(y_train))).ravel()
    # labels_pred = labels_pred.reshape((1, len(y_train))).ravel()

    # ari = metrics.adjusted_rand_score(y_train, labels_pred)
    # ami = metrics.adjusted_mutual_info_score(y_train, labels_pred)

    # print("ARI")
    # print(ari)
    # print("AMI")
    # print(ami)


# va modificato. accetta labels, eval_criterion e metodo di clustering. calcola il numero di cluster e procede
def clustering_evaluation(labels_true, labels_pred, evaluation_criterion='ari'):
    # Reshaping label vectors.
    labels_true = labels_true.reshape((1, len(labels_true))).ravel()
    labels_pred = labels_pred.reshape((1, len(labels_pred))).ravel()
    evaluation_criterion_dictionary = {
        'ari': metrics.adjusted_rand_score(labels_true, labels_pred),
        'ami': metrics.adjusted_mutual_info_score(labels_true, labels_pred)}
    score = evaluation_criterion_dictionary[evaluation_criterion]
    # 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_model", default='ResNet50')
    parser.add_argument("--pca_components", default=200)
    parser.add_argument("--do_pca", default=True)
    parser.add_argument("--do_kpca", default=True)
    parser.add_argument("--do_clustering", default=True)
    parser.add_argument("--do_tsne", default=True)
    parser.add_argument("--dataset", default='cifar10')
    parser.add_argument("--cluster_algorithm", default='KMeans')

    args = parser.parse_args()

    compute_bottleneck_features(args)