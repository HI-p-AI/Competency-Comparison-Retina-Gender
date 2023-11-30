# Setting Tensorflow log level to ignore unnecessery logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Logging Tensorflow version and GPU information
import tensorflow as tf
print(f'Tensorflow Version: {tf.__version__}')
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.experimental.get_device_details(gpus[0])
print(f'Device: {gpus["device_name"]}\nCompute Capability: {gpus["compute_capability"]}')


# Importing necessary libraries
from tensorflow import keras
from config import Config
from models import get_model
from dataset import LoadDataset
from augmentations import ImageAugmentation

# Setting random seed for tensorflow, python and numpy
SEED = 225
keras.utils.set_random_seed(SEED)


if __name__ == '__main__':
    config = Config()
    config.update_config({'model_name': 'vgg19', 'seed': SEED})
    config = config()
    
    
    train_ds, val_ds, test_ds = LoadDataset(config)()
    augmented_train_ds = ImageAugmentation(config).augment(train_ds)
    
    
    model, preprocess = get_model(config)
    
    base_model = model.layers[1]
    base_model.trainable = False
    
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=config['loss_fn'],
        metrics=[
            config['main_metric'],
        ],
    )
    
    history = model.fit(augmented_train_ds, epochs=config['last_layers_training_epochs'], validation_data=val_ds)
    