# ==============================================
#     This file defines the way to get the model 
#     from the configuration.
#
#
#     Md. Ahanaf Arif Khan
#     Contact: ahanaf019@gmail.com
# ==============================================


# Inporting Dependencies
import tensorflow as tf
from tensorflow import keras
from config import Config


def get_base_model(config: dict):
    """
    Function to get the base model and preprocessing function for that
    model.

    Args:
        config (dict): Configuration file for the run.

    Returns:
        keras.Model: The desired base model
        Callable: Preprocessing function for that model
    """
    
    # Extracting parameters from config
    model_name = config['model_name']
    args_dict = {
        'include_top': False,
        'input_shape': config['image_size'] + (3,),
        'weights': config['init_weights']
    }
    
    models = {
        'densenet201': keras.applications.DenseNet121(**args_dict),
        'inception_v3': keras.applications.InceptionV3(**args_dict),
        'inception_resnet_v2': keras.applications.InceptionResNetV2(**args_dict),
        'resnet152': keras.applications.ResNet152(**args_dict),
        'vgg16': keras.applications.VGG16(**args_dict),
        'vgg19': keras.applications.VGG19(**args_dict),
    }
    
    preprocess = {
        'densenet201': keras.applications.densenet.preprocess_input,
        'inception_v3': keras.applications.inception_v3.preprocess_input,
        'inception_resnet_v2': keras.applications.inception_resnet_v2.preprocess_input,
        'resnet152': keras.applications.resnet.preprocess_input,
        'vgg16': keras.applications.vgg16.preprocess_input,
        'vgg19': keras.applications.vgg19.preprocess_input,
    }
    return models[model_name], preprocess[model_name]


def get_model(config: dict):
    """Create the model to use from config.

    Args:
        config (dict): Configuration file for the run

    Returns:
        keras.Model: The model to be trained
        Callable: the preprocessing function for that model
    """
    base_model, preprocess = get_base_model(config)
    image_shape = config['image_size'] + (3,)
    num_classes = config['num_classes']
    
    inputs = keras.Input(shape=image_shape)
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs), preprocess