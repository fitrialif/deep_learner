import numpy as np

from .network_utils import load_base_network
from .data_utils import load_keras_dataset


def compute_bottleneck_features(bottleneck_features_filename, network_model='VGG16', dataset='cifar10', input_shape=(197, 197, 3)):
    """ Compute bottleneck features.

    # Arguments:
        bottleneck_features_filename (string): The name of the file in which bottleneck features will be saved.
        network_model (string): The name of the network model.
        dataset (string): The name of the dataset.
        input_shape (3d tensor): The input shape of the network model.
    
    # Return:
        bottleneck_features (np.array): A numpy array which stores bottleneck features.
    """
    (x_train, y_train), (x_test, y_test) = load_keras_dataset(dataset)
    # y_train = np.squeeze(y_train)
    network_model = load_base_network(network_model, input_shape)

    if os.path.exists(bottleneck_features_filename):
        bottleneck_features = np.load(bottleneck_features_filename)['bottleneck_features']
    else:
        x_train_array = np.array([scipy.misc.imresize(x_train[i], input_shape) for i in range(0, len(x_train))]).astype('float32')
        if model_choice == 'InceptionV3':
            input_train = applications.inception_v3.preprocess_input(x_train_array)
        elif model_choice == 'ResNet50':
            input_train = applications.resnet50.preprocess_input(x_train_array)
        elif:
            input_train = x_train_array
        bottleneck_features = model.predict(input_train)
        # bottleneck_features = np.squeeze(bottleneck_features)
        np.savez(bottleneck_features_filename, bottleneck_features=bottleneck_features)
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