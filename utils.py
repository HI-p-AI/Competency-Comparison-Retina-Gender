import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


def initialization(config: dict):
    base_path = config['base_path']
    model_name = config['model_name']
    train_path = os.path.join(base_path, model_name)
    
    if os.path.exists(train_path):
        print('\033[93m' + 'Previous training data found. Would you like to delete it? (Or abort training)' + '\033[0m')
        x = input('Enter Response(y/n): ')
        
        if x.lower() != 'y':
            print('Aborted!')
            exit(1)
        shutil.rmtree(train_path)
    os.mkdir(train_path)
    
    filenames = {
        'fig_before_aug' : os.path.join(train_path, 'dataset_samples_before_aug.png'),
        'fig_after_aug' : os.path.join(train_path, 'dataset_samples_after_aug.png'),
        'history' : os.path.join(train_path, f'{model_name}_history'), # extention is handled inside plot_history function
        'model_summary' : os.path.join(train_path, f'{model_name}_model_summary.txt'),
        'full_training' : os.path.join(train_path, f'{model_name}_full_trained.h5'),
        # path for best checkpoint is defined in callbacks
    }
    return filenames


def plot_ds_samples(train_ds, title, class_names, save_filename):
    fig = plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(8):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    fig.suptitle(title, fontweight='bold', fontsize=20)
    plt.savefig(save_filename)
    plt.close()


def plot_history(history, filename, metrics=['loss', 'accuracy']):
    
    history: dict = history.history
    
    for metric in metrics:
        fig = plt.figure(figsize=(10, 10))
        f = f'{filename}_{metric}.png'
        plt.plot(history[metric], linestyle='dashdot', color='black')
        plt.plot(history[f'val_{metric}'], linestyle='dashdot', color='green')
        
        plt.legend([metric, f'val_{metric}'])
        plt.savefig(f)
        plt.close()