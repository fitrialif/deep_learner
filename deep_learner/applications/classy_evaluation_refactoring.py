from ..utils.learning_utils import compute_bottleneck_features, 

# TODO: refactor, is a sample
def classy_evaluation(args):
    network_model = args.network_model
    dataset = args.dataset
    perform_pca = args.perform_pca


    bottleneck_features_filename = network_model + '_bottleneck_features.npz'
    # Step 1: compute bottleneck features.
    compute_bottleneck_features(bottleneck_features_filename, network_model, dataset)
    # Step 2 (optional): compute PCA.
    if perform_pca:
        
    # Step 3 (optional): visualize data with t-SNE.

    # Step 4: clustering + evaluate clustering results (optional).


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bottleneck_features_filename', default)
    parser.add_argument('--network_model', default='VGG16')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument("--perform_pca", default=True)
    parser.add_argument("--pca_components", default=200)
    parser.add_argument("--do_clustering", default=True)
    parser.add_argument("--do_tsne", default=True)
    parser.add_argument("--dataset", default='cifar10')
    parser.add_argument("--clustering_algorithm", default='KMeans')

    args = parser.parse_args()

    classy_evaluation(args)