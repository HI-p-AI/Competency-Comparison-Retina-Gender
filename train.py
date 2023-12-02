from tensorflow import keras
from callbacks import get_callbacks


def train_model(model: keras.Model,  config:dict, train_ds, val_ds, epochs:int):
    """Function for compiling and training a model 

    Args:
        model (keras.Model): A keras model
        config (dict): The configuration dictionary.
        epochs (int, optional): Number of epochs to train for.
        train_ds: The training dataset.
        val_ds: The validation dataset.

    Returns:
        Returns the history object after training the model.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=config['loss_fn'],
        metrics=[
            config['main_metric'],
        ],
    )
    
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=val_ds,
        initial_epoch=0, 
        callbacks=get_callbacks(config),
        verbose=1
    )
    
    return history