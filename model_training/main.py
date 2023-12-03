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
from contextlib import redirect_stdout
from tensorflow import keras
from config import Config
from models import get_model
from dataset import LoadDataset
from augmentations import ImageAugmentation
from train import train_model
from utils import plot_ds_samples, initialization, plot_history

# Setting random seed for tensorflow, python and numpy
SEED = 225
keras.utils.set_random_seed(SEED)
    
def main():
# Update and load config
    config = Config()
    config.update_config({
        'model_name': 'vgg19', 
        'seed': SEED,
        'image_size': (256, 256),
        'batch_size': 32
    })
    config = config()
    
    # Initialize the training here
    filenames = initialization(config)
    
    # Get model and preprocess function for that model
    model, preprocess_function = get_model(config)
    
    # Load the dataset and plot samples
    dataset = LoadDataset(config)
    train_ds, val_ds, test_ds = dataset()
    class_names = dataset.get_class_names()
    plot_ds_samples(train_ds, 'Train Dataset Samples Before Augmentation', class_names, filenames['fig_before_aug'])
    
    
    # Augment the train dataset and plot samples
    augmented_train_ds = ImageAugmentation(config).augment(train_ds)
    plot_ds_samples(augmented_train_ds, 'Train Dataset Samples After Augmentation', class_names, filenames['fig_after_aug'])
    
    # Apply preprocessing function on augmented train, validation and test sets
    augmented_train_ds = augmented_train_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Freeze the layers of the base model
    base_model = model.layers[1]
    base_model.trainable = False
    
    # Saving model summary inside txt file
    with open(filenames['model_summary'], 'w') as f:
        with redirect_stdout(f):
            model.summary(expand_nested=True, show_trainable=True)

    # Train the newly added layers of the model
    history = train_model(model, config, augmented_train_ds, val_ds, config['last_layers_training_epochs'])

    # Fine-tune the entire model
    base_model.trainable = True
    history = train_model(model, config, augmented_train_ds, val_ds, config['full_model_fine_tuning_epochs'])
    model.save(filenames['full_training'])
    plot_history(history, filenames['history'])
    
    
    
if __name__ == '__main__':
    main()