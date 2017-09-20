import keras
import numpy as np
import scipy
from scipy import misc
import os

import os
import sys
import glob
import argparse
from .utils import utils

from keras import applications
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import matplotlib.cm as cm

n=1

# Parameters
img_width, img_height = 75*n, 187*n


def train(args):
    """
    Performs training.
    """
    train_dir = args.train_folder
    nb_train_samples = utils.get_nb_files(args.train_folder)
    nb_classes = utils.get_labels(args.train_folder, args.validation_folder)
    base_architecture = args.base_architecture
    # Define base layer
    if base_architecture == 'VGG16':
        model = applications.VGG16(weights='imagenet', include_top=False)
    elif base_architecture == 'VGG19':
        model = applications.VGG19(weights='imagenet', include_top=False)
    elif base_architecture == 'InceptionV3':
        model = applications.InceptionV3(weights='imagenet', include_top=False)
    elif base_architecture == 'ResNet50':
        model = applications.ResNet50(weights='imagenet', include_top=False)
    
    data_gen = ImageDataGenerator(rescale = 1./255)

    data_generator = data_gen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=16,
        class_mode='categorical'
    )

    predictions = model.predict_generator(data_generator, val_samples=1000)

    predictions = np.squeeze(predictions)
    # np.savez('inception_features', predictions=predictions)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    
    
    # Reshape the array.
    nsamples, nx, ny = predictions.shape
    res_predictions = predictions.reshape((nsamples, nx*ny))
    n_clusters = int(args.n_clusters)

    ax1.set_ylim([0, len(res_predictions) + (n_clusters + 1) * 10])

    
    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans = clusterer.fit_predict(res_predictions)
    
    silhouette_avg = silhouette_score(res_predictions, kmeans)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(res_predictions, kmeans)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[kmeans == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(kmeans.astype(float) / n_clusters)
    ax2.scatter(res_predictions[:, 0], res_predictions[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.show()

if __name__=='__main__':
    # Callbacks array
    tensorboard = TensorBoard(log_dir='./logs')
    early = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=3, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint("weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [tensorboard, early, checkpoint]
    a = argparse.ArgumentParser()
    a.add_argument("--base_architecture", default='InceptionV3')
    a.add_argument("--train_folder", default='./data/train')
    a.add_argument("--validation_folder", default='./data/test')
    a.add_argument("--nb_epoch", default=1)
    a.add_argument("--batch_size", default=128)
    a.add_argument("--output_model_file", default="inceptionv3.model")
    a.add_argument("--plot", action="store_true")
    a.add_argument("--model_load", default='')
    a.add_argument("--callbacks", default = callbacks)
    a.add_argument("--n_clusters", default=2)

    args = a.parse_args()

    if args.train_folder is None or args.validation_folder is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_folder)) or (not os.path.exists(args.validation_folder)):
        print("directories do not exist")
        sys.exit(1)

    train(args)