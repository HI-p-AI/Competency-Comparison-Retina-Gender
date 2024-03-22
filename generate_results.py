import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_training.models import get_preprocess


def get_img_array(img_path, size, preprocess):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess(array)
    return array


image_size = (256, 256)
save_path = './saves'
dataset_base_path = './RU_RetinaDB_Gender'
model_names = [
    'densenet201', 
    'inception_resnet_v2', 
    'vgg16', 
    'vgg19', 
    'inception_v3', 
    'resnet152'
]


male_images = sorted(glob(f'{dataset_base_path}/Test/M/*'))
female_images = sorted(glob(f'{dataset_base_path}/Test/F/*'))
all_images = male_images + female_images
types = [ 'male' for i in range(len(male_images)) ] + [ 'female' for i in range(len(female_images)) ]
all_image_names = [ x.split('/')[-1] for x in all_images ]
print(f'Test Dataset:')
print(f'Male Images:\t{len(male_images)}')
print(f'Female Images:\t{len(female_images)}')
print(f'Total Images:\t{len(all_images)}')


results = []
pred_index = {
    'male': 1,
    'female': 0
}

for ext in ['_best_checkpoint', '_full_trained']:
    for i in range(len(model_names)):
        model_path = os.path.join(save_path, model_names[i], f'{model_names[i]}{ext}.h5')
        model = keras.models.load_model(model_path)
        preprocess = get_preprocess(model_names[i])
        print(f'{model_names[i]}{ext} Loaded Successfully')
        
        predictions = []
        j = 0
        for img in tqdm(all_images):
            image = get_img_array(img, image_size, preprocess)
            pred = model.predict(image, verbose=0)
            predictions.append(pred[0][pred_index[types[j]]])
            j += 1
        results.append(predictions)


    dictionary = {
        'filename' : all_image_names,
        'type' : types,
        'densenet201' : results[0], 
        'inception_resnet-v2' : results[1],
        'vgg16' : results[2],
        'vgg19' : results[3],
        'inception-v3' : results[4],
        'resnet152' : results[5],
    }

    df = pd.DataFrame(dictionary)
    df.to_csv(f'./results/results{ext}.csv', index=False)