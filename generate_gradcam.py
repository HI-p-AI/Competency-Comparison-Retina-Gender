# SOURCE: https://keras.io/examples/vision/grad_cam/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
from glob import glob
import numpy as np
import os
from tqdm import tqdm
from model_training.models import get_preprocess
import shutil
# Display
from IPython.display import Image, display
import matplotlib.cm as cm


image_size = (256, 256)
save_path = './saves'
output_base_dir = './results/heatmaps'
dataset_base_path = './RU_RetinaDB_Gender'
model_names = [
    'densenet201', 
    'inception_resnet_v2', 
    'vgg16', 
    'vgg19', 
    'inception_v3', 
    'resnet152'
]

pred_index = {
    'female': 0,
    'male': 1,
}


def main():
    male_images = sorted(glob(f'{dataset_base_path}/Test/M/*'))
    female_images = sorted(glob(f'{dataset_base_path}/Test/F/*'))
    all_images = [female_images] + [male_images]
    print(f'Test Dataset:')
    print(f'Male Images:\t{len(male_images)}')
    print(f'Female Images:\t{len(female_images)}')
    
    
    for ext in ['_best_checkpoint', '_full_trained']:
        for i in range(len(model_names)):

            model_path = os.path.join(save_path, model_names[i], f'{model_names[i]}{ext}.h5')
            preprocess = get_preprocess(model_names[i])
            model, last_conv_layer_name = get_model_and_last_conv_layer_name(model_path)
            model.layers[-1].activation = None
            print(f'{model_names[i]}{ext} Loaded')
            # ===========================================================================
            
            output_dir = os.path.join(output_base_dir, f'{model_names[i]}{ext}')
            os.mkdir(output_dir)
            
            overlay_dir_m = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'overlays_male')
            heat_dir_m = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'heatmaps_male')
            os.mkdir(overlay_dir_m)
            os.mkdir(heat_dir_m)
            
            overlay_dir_f = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'overlays_female')
            heat_dir_f = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'heatmaps_female')
            os.mkdir(overlay_dir_f)
            os.mkdir(heat_dir_f)
            
            overlay_dirs = [overlay_dir_f, overlay_dir_m]
            heat_dirs = [heat_dir_f, heat_dir_m]
            
            for idx in pred_index.keys():
                k = pred_index[idx]
                print(f'Generating for {idx} images:')
                for img_path in tqdm(all_images[k]):
                    img_array = preprocess(get_img_array(img_path, size=image_size))
                    filename = os.path.join(overlay_dirs[k], img_path.split('/')[-1])
                    heatname = os.path.join(heat_dirs[k], img_path.split('/')[-1])

                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=k)
                    save_and_display_gradcam(img_path, heatmap, cam_path=filename, heat_path=heatname, alpha=0.4)


def get_model_and_last_conv_layer_name(model_path):
    model = keras.models.load_model(model_path)

    backbone = model.layers[1]
    inp = backbone.input

    x = backbone.output
    x = model.layers[2](x)
    x = model.layers[3](x)
    x = model.layers[4](x)
    x = model.layers[5](x)
    x = model.layers[6](x)

    pmodel = keras.Model(inp, x)
    last_conv_layer_name = pmodel.layers[-6].name
    return pmodel, last_conv_layer_name


def get_img_array(img_path, size):
    # `img` is a PIL image of target size
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array
    array = keras.utils.img_to_array(img)
    # Add a dimension to transform our array into a "batch"
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model: keras.Model, last_conv_layer_name: str, pred_index:int=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", heat_path='heat.jpg', alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    
    jet_heatmap.save(heat_path)
    
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    

if __name__ == '__main__':
    if os.path.exists(output_base_dir):
        print('\033[93m' + 'Previous data found. Would you like to delete it? (Or abort training)' + '\033[0m')
        x = input('Enter Response(y/n): ')
        
        if x.lower() != 'y':
            print('Aborted!')
            exit(1)
        shutil.rmtree(output_base_dir)
    
    os.mkdir(output_base_dir)
    
    # Run the main function
    main()