# ==============================================
#     Experiment configurations
#
#
#     Md. Ahanaf Arif Khan, 2023
#     Contact: ahanaf019@gmail.com
# ==============================================

from typing import Any
from tensorflow import keras

class Config:
    """Class for containing all the configurations of the experiments.
    
    Configurations include hyperparameters, model name, dataset path, epochs to train, etc.
    """
    __config = {
        'model_name': 'vgg16',
        'image_size': (256, 256),
        'init_weights': 'imagenet',
        'batch_size': 32,
        'label_mode': 'categorical',
        'base_path': './training',
        'last_layers_training_epochs': 20,
        'full_model_fine_tuning_epochs': 150,
        'early_stop_patience': 60,
        'reduce_lr_patience': 25,
        'num_classes': 2,
        'loss_fn': keras.losses.CategoricalCrossentropy(),
        'main_metric': keras.metrics.CategoricalAccuracy(),
        'monitor_metric': 'val_loss',
        'seed': 225,
        'dataset_dir': '/media/ahanaf/media-1/Datasets/RU_Retina_Gender_Database_Norm',
    }
    
    
    def __call__(self) -> dict:
        """Get configuration file in its current form

        Returns:
            dict: the configuration dict
        """
        
        return self.__config
    
    
    def update_config(self, update_dict: dict):
        """Update configuration with a given dictionary

        Args:
            update_dict (dict): dictionary to update from
        """
        self.__config.update(update_dict)