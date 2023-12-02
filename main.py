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
from train import train_model

# Setting random seed for tensorflow, python and numpy
SEED = 225
keras.utils.set_random_seed(SEED)
    
def main():
# Update and load config
    config = Config()
    config.update_config({
        'model_name': 'vgg19', 
        'seed': SEED
    })
    config = config()
    
    # Get model and preprocess function for that model
    model, preprocess_function = get_model(config)
    
    # Load the dataset
    train_ds, val_ds, test_ds = LoadDataset(config)()
    
    # Augment the train dataset
    augmented_train_ds = ImageAugmentation(config).augment(train_ds)
    
    # Apply preprocessing function on augmented train, validation and test sets
    augmented_train_ds = augmented_train_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Freeze the layers of the base model
    base_model = model.layers[1]
    base_model.trainable = False
    
    model.summary()
    # Train the newly added layers of the model
    history = train_model(model, config['last_layers_training_epochs'])
    
    # Fine-tune the entire model
    base_model.trainable = True
    history = train_model(model, config['full_model_fine_tuning_epochs'])
    
    # model.save(FINAL_MODEL_SAVE_PATH)
    
    
if __name__ == '__main__':
    main()