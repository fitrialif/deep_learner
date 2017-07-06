from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


def setup_transfer_learn(base_model, model, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']):
    """"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def setup_fine_tuning(model, layers_to_freeze=0, optimizer=optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']):
    """"""
    for layer in model.layers[:layers_to_freeze]:
        layer.trainable = False
    for layer in model.layers[layers_to_freeze:]:
        layer.trainable = True
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def replace_classification_layer(base_model, fc_size, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model