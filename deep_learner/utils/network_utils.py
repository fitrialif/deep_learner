from keras.applications import VGG16, VGG19, InceptionV3, ResNet50, Xception


__network_model_dictionary = {
        'Xception': Xception(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'VGG19': VGG19(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=input_layer_shape),
        'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=input_layer_shape)}


def load_base_network(network_model='VGG16', input_layer_shape=(197, 197, 3)):
    """Load a pretrained base network model.

    Args:
        network_model (string): A string which represents the base network model.
        input_layer_shape (3d tensor): A 3d tensor (height x width x channels) which represents the input layer's shape.
    
    Returns:
        model: the base model for the selected network.
    """
    if network_model not in __network_model_dictionary:
        raise ValueError('Could not find a valid network model for argument &s.' % (network_model))
    
    return __network_model_dictionary[network_model]