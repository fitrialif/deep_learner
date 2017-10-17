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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, DBSCAN, Birch
from sklearn import metrics


def load_base_network(network_model='VGG16', input_layer_shape=(197, 197, 3)):
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


def load_dataset(dataset='cifar10'):
    dataset_dictionary = {
        'cifar10': cifar10.load_data(),
        'cifar100': cifar100.load_data()}
    return dataset_dictionary[dataset]


def choose_clustering_algorithm(clustering_algorithm='KMeans', n_clusters=10):
    clustering_dictionary = {
        'KMeans': KMeans(n_clusters=n_clusters),
        'MiniBatchKMeans': MiniBatchKMeans(n_clusters=n_clusters),
        'AffinityPropagation': AffinityPropagation()}
    return clustering_dictionary[clustering_algorithm]


# TODO: REFACTORING NEEDED
def classy_evaluation(args):
    """
    Main method for the file.
    """
    dataset = args.dataset
    network_model = args.network_model
    perform_pca = args.perform_pca

features, pca_filename, pca_components=200, pca_savefig=True

    # Step 1. Compute bottleneck features.
    bottleneck_features_filename = network_model + '_bottleneck_features.npz'
    bottleneck_features = compute_bottleneck_features(bottleneck_features_filename, network_model, dataset)
    # Step 2. Perform PCA (if specified).
    if perform_pca == True:
        pca_components = compute_pca_components(features)
        


    clusters_filename = clustering_algorithm + '_predicted_clusters.npz'

# REFACTORING NEEDED: PASS FEATURE NAME
def compute_bottleneck_features(bottleneck_features_filename, model_choice='VGG16', dataset='cifar10', input_shape=(197, 197, 3)):
    """
    Compute bottleneck features.
    Arguments:
        model_choice:
        dataset:
        input_shape:
    Return:
        bottleneck_features:
    """
    # Load dataset and network model.
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset)
    y_train = np.squeeze(y_train)
    print('Dataset loaded.')
    model = load_network(model_choice)
    print('Pretrained network model loaded.')

    # Load or compute bottleneck features. If the feature filename exists, load it. Otherwise, compute features.
    if os.path.exists(bottleneck_features_filename):
        print('Bottleneck features detected. Loading...')
        features = np.load(bottleneck_features_filename)['features']
    else:
        # Pre-process data. Fit them into a Numpy array, and pre-process them as needed.
        print('Bottleneck features not detected. Pre-processing data...')
        x_train_array = np.array([scipy.misc.imresize(x_train[i], input_shape) for i in range(0, len(x_train))]).astype('float32')
        if model_choice == 'InceptionV3':
            input_train = applications.inception_v3.preprocess_input(x_train_array)
        elif model_choice == 'ResNet50':
            input_train = applications.resnet50.preprocess_input(x_train_array)
        elif:
            input_train = x_train_array
        print('Pre-processing complete. Predicting bottleneck features...')
        bottleneck_features = model.predict(input_train)
        bottleneck_features = np.squeeze(bottleneck_features)
        print('Bottleneck features predicted. Saving...')
        np.savez(bottleneck_features_filename, features=bottleneck_features)
        print('Bottleneck features saved.')    
    return features

def compute_clusters(features, clusters_filename, clustering_algorithm='KMeans'):
    """
    Compute clusters.
    Arguments:
        features:
        clusters_filename:
        clustering_algorithm:
    Return:
        predicted_clusters:
    """
    if os.path.exists(clusters_filename):
        print('Predicted clusters detected. Loading...')
        predicted_clusters = np.load(clusters_filename)['predicted_clusters']
    else:
        print('Predicted clusters not detected. Predicting clusters...')
        clustering_model = choose_clustering_algorithm(clustering_algorithm)
        predicted_clusters = clustering_model.fit_predict(features.reshape([features.shape[0], np.prod(features.shape[1:])]).astype('float32'))
        np.savez(clusters_filename, predicted_clusters = predicted_clusters)
    return predicted_clusters

def compute_pca_components(features, pca_filename, pca_components=200, pca_savefig=True):
    """
    Compute PCA starting from a set of features.
    Arguments:
        features: the set of features on which PCA must be computed.
        pca_components: the number of principal components which should be considered.
        pca_savefig: a flag which states whether a plot of the cumulative explained variance should be saved or not.
    Returns:
        pca_components: the set of computed PCA components.
    """
    pca_filename = model_choice + '_pca_components.npz'
    if os.path.exists(pca_filename):
        print('loading pca')
        pca_features = np.load(pca_file)['pca_components']
        explained_variance = np.load(pca_file)['explained_variance']
    else:
        print('computing pca features')
        pca = PCA(n_components=200)
        pca_features = pca.fit_transform(features.reshape([features.shape[0], np.prod(features.shape[1:])]).astype('float'))
        explained_variance = pca.explained_variance_ratio_
        np.savez(pca_file, pca_features=pca_features, explained_variance=explained_variance)
        print('pca saved')

    if pca_savefig == True:        
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('Pca components')
        plt.ylabel('Explained variance (%)')
        plt.title('Variance explained by PCA')
        plt.savefig(model_choice + '_pca_explained_variance.jpg')

    return pca_components

# REFACTORING NEEDED
def compute_tsne_features(data):
    tsne_filename = 

    if do_tsne == True:
        tsne_file = model_choice + '_tsne_features.npz'
        if os.path.exists(tsne_file):
            print('load tsne features')
            tsne_features = np.load(tsne_file)['tsne_features']
        else:
            print('tsne fatures not detected (test)')
            print('calculating now...')
            tsne_features = TSNE(n_components=2).fit_transform(pca_features.reshape([pca_features.shape[0], np.prod(pca_features.shape[1:])]).astype('float')[:25000])
            print('saving...')
            np.savez(tsne_file, tsne_features=tsne_features)
            print('tsne feature saved')
        plt.figure(figsize=(20,20))
        plt.scatter(tsne_features[:,0], tsne_features[:,1], c=plt.cm.jet(y_train/10), s=10, edgecolors='none')
        plt.title('t-SNE clusters')
        plt.savefig(model_choice + '_tsne.jpg')


def iterative_evaluate_clusters(labels, predicted_clusters, metrics='ARI', k=5):
    """
    Iteratively evaluate clustering results.
    Arguments:
        labels:
        predicted_clusters:
        metrics:
    Returns:

    """
    # Pre-process labels and predicted_clusters vectors.
    labels = labels.reshape((1, len(labels))).ravel()
    predicted_clusters = predicted_clusters.reshape((1, len(predicted_clusters))).ravel()

    metric_dictionary = {
        'ARI': metrics.adjusted_rand_score(labels, predicted_clusters),
        'AMI': metrics.adjusted_mutual_info_score(labels, predicted_clusters)
    }
    # TODO: RETURN A LIST
    # TODO: ITERATE THIS  

    return metric_dictionary[metric]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_model", default='VGG16')
    parser.add_argument("--perform_pca", default=True)
    parser.add_argument("--pca_components", default=200)
    parser.add_argument("--do_clustering", default=True)
    parser.add_argument("--do_tsne", default=True)
    parser.add_argument("--dataset", default='cifar10')
    parser.add_argument("--clustering_algorithm", default='KMeans')

    args = parser.parse_args()

    compute_bottleneck_features(args)