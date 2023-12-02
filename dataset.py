import tensorflow as tf
from tensorflow import keras
import os

class LoadDataset:
    def __init__(self, config: dict) -> None:
        self.seed = config['seed']
        self.image_size = config['image_size']
        self.batch_size = config['batch_size']
        self.label_mode = config['label_mode']
        self.dataset_dir = config['dataset_dir']


    def __call__(self) -> tuple:
        train_set_dir = os.path.join(self.dataset_dir, 'Train')
        valid_set_dir = os.path.join(self.dataset_dir, 'Validation')
        test_set_dir = os.path.join(self.dataset_dir, 'Test')
        
        
        train_ds = keras.utils.image_dataset_from_directory(
            train_set_dir,
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode=self.label_mode,
        )

        validation_ds = keras.utils.image_dataset_from_directory(
            valid_set_dir,
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode=self.label_mode,
        )

        test_ds = keras.utils.image_dataset_from_directory(
            test_set_dir,
            seed=self.seed,
            image_size=self.image_size,
            batch_size=None,
            label_mode=self.label_mode,
        )
        self.class_names = validation_ds.class_names
        train_data = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        val_data = validation_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        test_data = test_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
        print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))
        print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))
        print(f'Class Names(indexed): {validation_ds.class_names}')
        
        return train_data, val_data, test_data
    
    
    def get_class_names(self):
        return self.class_names