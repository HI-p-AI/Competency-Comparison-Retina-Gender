from typing import Any
from tensorflow import keras

class Config:
    __config = {
        'model_name': 'vgg16',
        'image_size': (256, 256),
        'init_weights': 'imagenet',
        'batch_size': 32,
        'label_mode': 'categorical',
        'base_path': '.',
        'checkpoint_path': 'checkpoints',
        'full_model_path': 'full_trained',
        'last_layers_training_epochs': 20,
        'full_model_fine_tuning_epochs': 150,
        'early_stop_patience': 60,
        'num_classes': 2,
        'loss_fn': keras.losses.CategoricalCrossentropy(),
        'main_metric': keras.metrics.CategoricalAccuracy(),
        'monitor_metric': 'val_loss',
        'seed': 225,
        'dataset_dir': '/media/ahanaf/media-1/Datasets/RU_Retina_Gender_Database_Norm',
        
        
        
    }
    
    def __call__(self) -> dict:
        return self.__config
    
    
    def update_config(self, update_dict):
        self.__config.update(update_dict)