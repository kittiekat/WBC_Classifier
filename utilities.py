import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as s
import tensorflow as tf
import os, random
from PIL import Image

def plot_results(model, epochs, title):
    # Graph of Accuracy
    acc_train = model.history['accuracy']
    acc_val = model.history['val_accuracy']
    plt.plot(range(1, epochs + 1), acc_train, 'purple', label='Training accuracy')
    plt.plot(range(1, epochs + 1), acc_val, 'violet', label='Validation accuracy')
    plt.title(title + ' Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Graph of Loss
    loss_train = model.history['loss']
    loss_val = model.history['val_loss']
    plt.plot(range(1, epochs + 1), loss_train, 'purple', label='Training loss')
    plt.plot(range(1, epochs + 1), loss_val, 'violet', label='Validation loss')
    plt.title(title + ' Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def cm(y_pred, y_truth, classes):
    # Confusion Matrix https://androidkt.com/keras-confusion-matrix-in-tensorboard/

    con_mat = tf.math.confusion_matrix(labels=y_truth, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    figure = plt.figure(figsize=(8, 8))
    s.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def max_count(my_dir):
    cat_counts = []
    my_dirs = os.listdir(my_dir)

    for sub_dir in my_dirs:
        cat_counts.append(sum([len(files) for r, d, files in os.walk(my_dir + "/" + sub_dir)]))
    cat_count = max(cat_counts)

    print("The maximum category count is:",cat_count)
    return cat_count

def augment(my_dir, max_count):
    from distutils.dir_util import copy_tree

    # Create a new directory for the data
    new_directory = my_dir[:-1] + " - Augmented/"
    copy_tree(my_dir, new_directory)

    my_dirs = os.listdir(new_directory)

    for sub_dir in my_dirs:

        while sum(len(files) for _, _, files in os.walk(new_directory+sub_dir)) > 0 and sum(len(files) for _, _, files in os.walk(new_directory+sub_dir)) < max_count:
            # Pick image
            file_name =  random.choice(os.listdir(my_dir + sub_dir))
            picked_image = Image.open(my_dir + sub_dir + "/" + file_name)

            # Rotate image
            picked_image = picked_image.rotate(random.randint(5, 85))

            # Pick file name
            new_name = new_directory + sub_dir + "/" + file_name[0:-4] + "-flip-" + str(random.randint(0,99999)) + file_name[-4:]
            while os.path.exists(new_name):
                new_name = new_directory + sub_dir + "/" + file_name[0:-4] + "-flip-" + str(random.randint(0,99999)) + file_name[-4:]

            # Flip and save image
            img = np.array(picked_image)
            i = random.randint(0, 2)
            if i == 0:
                Image.fromarray(np.flip(img, (0, 1))).save(new_name)
            elif i == 1:
                Image.fromarray(np.flip(img, 0 )).save(new_name)
            else:
                Image.fromarray(np.flip(img, 1)).save(new_name)

            print("File " + new_name + " has been created")


