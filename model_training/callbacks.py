# ==============================================
#     Callbacks definition
#
#
#     Md. Ahanaf Arif Khan, 2023
#     Contact: ahanaf019@gmail.com
# ==============================================

from tensorflow import keras
from config import Config
import os


def get_callbacks(config: dict):
    """Defines the callbacks used during model training.
    
    All callbacks have verbosity set to 1 (Log when activated)

    Args:
        config (dict): The config dictionary

    Returns:
        Returns a list containing the callbacks
    """
    
    # Get essential variables from config dict
    monitor_metric = config['monitor_metric']
    es_patience = config['early_stop_patience']
    reduce_lr_patience = config['reduce_lr_patience']
    checkpoint_save_path = os.path.join(
        config['base_path'], 
        config['model_name'], 
        f'{config["model_name"]}_best_checkpoint.h5'
    )
    
    # Callback definition
    es_callback = keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        patience=es_patience,
        verbose=1,
        mode="auto",
    )

    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_save_path,
        monitor = monitor_metric,
        save_best_only = True,
        save_weights_only= False,
        mode = "auto",
        verbose=1,
    )

    lr_callback = keras.callbacks.ReduceLROnPlateau(
        patience=reduce_lr_patience,
        verbose=1,
        min_lr=1e-7
    )

    return [
        cp_callback,
        lr_callback,
        es_callback,
    ]