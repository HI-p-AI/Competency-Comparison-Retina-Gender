import tensorflow as tf
from tensorflow import keras
import keras_cv
from config import Config

class ImageAugmentation:
    def __init__(self, config:Config) -> None:
        self.seed = config['seed']
        self.image_size = config['image_size']
        self.batch_size = config['batch_size']
        self.label_mode = config['label_mode']


    def augment(self, train_ds: tf.data.Dataset) -> tf.data.Dataset:
        
        random_zoom = keras.layers.RandomZoom(
            height_factor=(-0.05, 0.15), 
            seed=self.seed,
            fill_mode="nearest",
        )
        random_translation = keras.layers.RandomTranslation(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=self.seed,
        )
        random_flip = keras.layers.RandomFlip(
            mode="horizontal_and_vertical", 
            seed=self.seed
        )
        random_rotate = keras.layers.RandomRotation(
            factor=0.2, 
            fill_mode="nearest", 
            seed=self.seed,
        )
        random_cutout = keras_cv.layers.RandomCutout(
            height_factor=0.4, 
            width_factor=0.4, 
            seed=self.seed
        )
        random_shear = keras_cv.layers.RandomShear(
            x_factor=0.25,
            y_factor=0.25,
            interpolation="bilinear",
            fill_mode="reflect",
            seed=self.seed,
        )
        random_channel_shift = keras_cv.layers.RandomChannelShift(
            value_range=(0, 255),
            factor=0.5,
            seed=self.seed,
        )
        random_brightness = keras.layers.RandomBrightness(
            factor=0.2, 
            value_range=(0, 255), 
            seed=self.seed, 
        )
        random_contrast = keras.layers.RandomContrast(
            factor=0.4, 
            seed=None,
        )
        grayscale = keras_cv.layers.Grayscale(
            output_channels=3,
        )
        
        augmentation_list1 = [
            random_cutout,
            random_shear,
            random_channel_shift,
            grayscale,
        ]

        normal_augs = keras.Sequential([
            random_zoom, 
            random_translation,
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
        ])

        pipeline1 = keras_cv.layers.RandomAugmentationPipeline(
            layers=augmentation_list1, augmentations_per_image=1
        )


        def f(x, y):
            x = pipeline1(x)
            x = normal_augs(x)
            return x, y

        train_data = train_ds.map(lambda x, y: (f(x, y)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        print('Datasets Loaded With Augmentation')
        return train_data