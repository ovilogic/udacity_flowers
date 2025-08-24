import itertools
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import platform

from sklearn.metrics import confusion_matrix

import numpy as np

from flowers_cnn import flowers_preprocessed_train_ds as trained_dataset, flowers_cnn, flowers_preprocessed_validation_ds as validation_dataset


# Assuming we want to visualise not the raw image but the pre-processed image. See what our pre-processing does.
# We are also assuming that the pre-processing delivers a tf.data.Dataset object.
def visualize_cnn_images(dataset, num_images=5):
    """
    Visualizes images from a TensorFlow dataset.
    
    Args:
        dataset: A tf.data.Dataset object containing images.
        num_images: Number of images to visualize.
    """
    plt.figure(figsize=(10, 10)) # Set the figure size for better visibility. 10x10 inches is a good size for displaying multiple images.
    
    for i, (image, label) in enumerate(dataset.take(num_images)):
         # Print the shape of the first image in the batch to understand its dimensions.``
        plt.subplot(1, num_images, i + 1) # The first argument is the number of rows, the second is the number of columns, and the third is the index of the subplot.
        print(image[0].shape)
        plt.imshow(image[0].numpy().squeeze()) # imshow() displays the image. squeeze() removes single-dimensional entries from the shape of the array.
        plt.title(f'Label: {classes[label[0].numpy()]}')
        # plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualise_training(history_object):
    """
    Visualises accuracy and loss during training.
    """
    x_values = list(range(1, len(history_object.history['loss']) + 1))  # Get the number of epochs from the history object.
    print(f"Number of epochs: {len(x_values)}", x_values)  # Print the number of epochs for debugging. Number of epochs: 3 [0, 1, 2]
    plt.plot(x_values, history_object.history['loss'], label='Training Loss', color='red')
    plt.plot(x_values, history_object.history['val_loss'], label='Validation Loss', color='orange')
    plt.plot(x_values, history_object.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(x_values, history_object.history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss/Accuracy')
    plt.show()
######################################################################
# Create a confusion matrix to see which classes are confused with each other.
test_dataset = trained_dataset.take(1)  # Take a subset of the dataset for testing. The first batch is number 0.

#####################################################################
# Confusion matrix
"""
What a confusion matrix does

A confusion matrix is like a scoreboard that shows how your classifier’s predictions stack up against the truth, class by class:

Each row = actual class.

Each column = predicted class.

The diagonal values = correctly classified samples.

Off-diagonal values = misclassifications, telling you which classes your model is confusing with others.

Example for a 3-class problem:

             Predicted
           A   B   C
Actual A [ 8   1   0 ]
       B [ 2   5   1 ]
       C [ 0   2   9 ]


This means:

Class A: 8 correct, 1 misclassified as B.

Class B: 5 correct, 2 misclassified as A, 1 as C.

Class C: 9 correct, 2 misclassified as B.

It’s a richer picture than accuracy because it shows where the model stumbles.
"""

def confuse_flowers(dataset, model):
    """
    Creates a confusion matrix for the dataset using the trained model.
    
    Args:
        dataset: A tf.data.Dataset object containing images and labels.
        model: A trained Keras model.
    """
    # sklearn’s confusion_matrix function does not plot anything. It just returns a 2D NumPy array of integers
           
    y_true = []
    y_pred = []
    
    for images, labels in dataset:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    cm = confusion_matrix(y_true, y_pred)
    # cm is of type numpy.ndarray, a 2D array of integers. It's shape is [n_classes, n_classes].
    print("Confusion Matrix:\n", cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Flower Classification')
    plt.colorbar(im)
    tick_marks = np.arange(len(flowers_cnn.classes))
    plt.xticks(tick_marks, flowers_cnn.classes, rotation=45)
    plt.yticks(tick_marks, flowers_cnn.classes, rotation=45)
    # normalize the confusion matrix. 
    # This means that the values in each row will sum to 1.
    # Basically, these are percentages of how often a true class was predicted as each class.
    cm = cm.astype('float') # because we are going to do float division.
    sum_of_rows = cm.sum(axis=1)
    # sum_of_rows is a 1D array of shape [n_classes]. Each element is the sum of the corresponding row in cm.
    sum_of_rows = sum_of_rows[:, np.newaxis] # reshape to [n_classes, 1] so that we can do broadcasting.
    cm = cm / sum_of_rows  # Broadcasting. Each element in a row is divided by the sum of that row.
    print("Normalized Confusion Matrix:\n", cm)
   
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white"
        if float(cm[i, j]) < 0.5:
            color = "black"
        cm_cell = round(float(cm[i, j]), 4)
        # print(i, j, cm_cell, f"{cm_cell * 100:.2f}%")
        """
        :.2f → format the number as fixed-point (f) with 2 decimal places.
        .2 = two digits after the decimal.
        f = fixed-point notation (not scientific notation).
        """
        """
        In NumPy, cm[i, j] means:
        i = row index (true class).
        j = column index (predicted class).
        In Matplotlib’s coordinate system, the first argument to plt.text(x, y, ...) is:
        x = horizontal axis (columns).
        y = vertical axis (rows). 
        By default, imshow in matplotlib treats the top-left corner as (0,0), and then:
        x increases to the right (like usual)
        y increases downward (inverted compared to standard Cartesian graphs)
        That’s why your i (row index) corresponds to vertical downward steps, and your j (column index) corresponds to horizontal steps rightward.  
        """
        plt.text(j - 0.5, i, f"{cm_cell * 100:.2f}%", color=color, )
    plt.tight_layout(pad=5) # Adjust layout to prevent clipping of tick-labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    """
    If you give it a 2D array (like your confusion matrix cm of shape [n_classes, n_classes]), it maps the numbers → colors and draws a rectangular grid. That’s exactly what you want.

    If you give it a 3D array, it interprets it as an RGB (or RGBA) image, e.g. shape (height, width, 3) for RGB or (height, width, 4) for RGBA. That’s how you can pass an actual color image into imshow.

    If you give it a 1D array, it won’t work directly — you’d need to reshape it into 2D first.
    """

# visualize_cnn_images(trained_dataset) # Call the function to visualize images from the dataset.
# Train the model in windows, then visualise the training history.
# hist_obj = flowers_cnn.train_model(trained_dataset, validation_dataset, epochs=50) # Train the model for 10 epochs.
# print(len(hist_obj.history['loss']), len(hist_obj.history['accuracy']), len(hist_obj.history['val_loss']), len(hist_obj.history['val_accuracy']))


# visualise_training(hist_obj) # Call the function to visualize training history.

# validation_labels = []
# for images, labels in validation_dataset.take(1):  # Take one batch from the validation dataset.
    
#     validation_labels.extend(labels.numpy().tolist())  # Convert labels to integers for confusion matrix.
#     print(images.shape, images[0].shape, images[0])

# print(f"Validation labels: {validation_labels}", len(validation_labels))  # Print the validation labels and their count. 

if platform.system() == 'Windows':
    saved_dir = r'C:\python_work\tensorFlow\wsl_venv\Udacity\flowers\saved_models'
else:
    saved_dir = '/mnt/c/python_work/tensorFlow/wsl_venv/Udacity/flowers/saved_models'

loaded_model = tf.keras.models.load_model(os.path.join(saved_dir, 'flowers1.keras'))  # Load the trained model.
confuse_flowers(validation_dataset, loaded_model)  # Call the function to create a confusion matrix for the validation dataset, where the problem is.
