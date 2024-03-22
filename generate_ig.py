# SOURCE: https://keras.io/examples/vision/integrated_gradients/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras
import cv2
from glob import glob
import numpy as np
import shutil
from model_training.models import get_preprocess
from tqdm import tqdm



def read_image(img_path):
    img = keras.utils.load_img(img_path, target_size=image_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images


def compute_gradients(images, model, preprocess, target_class_idx):
    images = preprocess(images)
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(baseline, img, model, preprocess, target_class_idx, m_steps=50, batch_size=5):
    image = img.copy()
    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Collect gradients.
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(baseline, image, alpha_batch, model, preprocess, target_class_idx)
        gradient_batches.append(gradient_batch)

    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients
    return integrated_gradients


@tf.function
def one_batch(baseline, image, alpha_batch, model, preprocess, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                        image=image,
                                                        alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                        model=model, preprocess=preprocess, target_class_idx=target_class_idx)
    return gradient_batch


def plot_img_attributions(baseline, image, model, preprocess, target_class_idx, m_steps=tf.constant(50), cmap=None, overlay_alpha=0.4, ig_fname='foo.png', overlay_fname='foo.png'):
    img = image.copy()
    attributions = integrated_gradients(
        baseline=baseline,
        img=img, model=model, preprocess=preprocess,
        target_class_idx=target_class_idx,
        m_steps=m_steps
    )

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    # Save IG attribution mask
    x = attribution_mask.numpy()
    x = x / np.max(x)
    x = x * 255
    x = x.astype(np.uint8)
    x = cv2.applyColorMap(x, cmap)
    cv2.imwrite(ig_fname, x)
    
    # Save overlay
    overlay = x + 0.4 * img
    cv2.imwrite(overlay_fname, overlay)

    
    
    
image_size = (256, 256)
save_path = './saves'
output_base_dir = './results/ig'
dataset_base_path = './RU_RetinaDB_Gender'

model_names = [
    # 'densenet201', 
    # 'inception_resnet_v2', 
    'vgg16', 
    'vgg19', 
    # 'inception_v3', 
    # 'resnet152'
]

pred_index = {
    'female': 0,
    'male': 1,
}


male_images = sorted(glob(f'{dataset_base_path}/Test/M/*'))
female_images = sorted(glob(f'{dataset_base_path}/Test/F/*'))
all_images = [female_images] + [male_images]
print(f'Test Dataset:')
print(f'Male Images:\t{len(male_images)}')
print(f'Female Images:\t{len(female_images)}')






if os.path.exists(output_base_dir):
    print('\033[93m' + 'Previous data found. Would you like to delete it? (Or abort training)' + '\033[0m')
    x = input('Enter Response(y/n): ')
    
    if x.lower() != 'y':
        print('Aborted!')
        exit(1)
    shutil.rmtree(output_base_dir)

os.mkdir(output_base_dir)



for ext in ['_best_checkpoint', '_full_trained']:
    for i in range(len(model_names)):
        print(f'Loading {model_names[i]}{ext}')
        
        model_path = os.path.join(save_path, model_names[i], f'{model_names[i]}{ext}.h5')
        preprocess = get_preprocess(model_names[i])
        model = keras.models.load_model(model_path)
        # =======================================================================
        
        output_dir = os.path.join(output_base_dir, f'{model_names[i]}{ext}')
        os.mkdir(output_dir)
        
        overlay_dir_m = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'overlays_male')
        ig_dir_m = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'ig_male')
        os.mkdir(overlay_dir_m)
        os.mkdir(ig_dir_m)
        
        overlay_dir_f = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'overlays_female')
        ig_dir_f = os.path.join(output_base_dir, f'{model_names[i]}{ext}', 'ig_female')
        os.mkdir(overlay_dir_f)
        os.mkdir(ig_dir_f)
        
        overlay_dirs = [overlay_dir_f, overlay_dir_m]
        ig_dirs = [ig_dir_f, ig_dir_m]
        # ========================================================================
        
        
        for idx in pred_index.keys():
            k = pred_index[idx]
            print(f'Generating for {idx} images:')
            for img_path in tqdm(all_images[k]):


                image = img_path

                overlay_filename = os.path.join(overlay_dirs[k], img_path.split('/')[-1])
                igname = os.path.join(ig_dirs[k], img_path.split('/')[-1])
                # print(k, filename)

                img_paths = {
                    'Image': image,
                }
                
                
                img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}
                baseline = tf.zeros(shape=(256,256,3))

                # x = img_name_tensors['Male'][0].copy()

                plot_img_attributions(
                    image=img_name_tensors['Image'][0],
                    baseline=baseline, 
                    model=model, 
                    preprocess=preprocess,
                    target_class_idx=k,
                    m_steps=240,
                    cmap=cv2.COLORMAP_PLASMA,
                    overlay_alpha=0.3, 
                    ig_fname=igname,
                    overlay_fname=overlay_filename
                )