import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import numpy as np

from flowers_cnn import flowers_preprocessed_train_ds as trained_dataset, classes, flowers_cnn, flowers_preprocessed_validation_ds as validation_dataset


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


def confusion_matrix(dataset, model, num_images=5):
    """
    Creates a confusion matrix for the dataset using the trained model.
    
    Args:
        dataset: A tf.data.Dataset object containing images and labels.
        model: A trained Keras model.
    """
    cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)
    validation_labels = []
    for images, labels in dataset.take(-1):
        
        
        y_true = []
        y_pred = []
        
        for images, labels in dataset:
            predictions = model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(predictions, axis=1))
        
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)

# visualize_cnn_images(trained_dataset) # Call the function to visualize images from the dataset.
# Train the model in windows, then visualise the training history.
# hist_obj = flowers_cnn.train_model(trained_dataset, validation_dataset, epochs=50) # Train the model for 10 epochs.
# print(len(hist_obj.history['loss']), len(hist_obj.history['accuracy']), len(hist_obj.history['val_loss']), len(hist_obj.history['val_accuracy']))


# visualise_training(hist_obj) # Call the function to visualize training history.

validation_labels = []
for images, labels in validation_dataset.take(-1):
    validation_labels.extend(labels.numpy().tolist())  # Convert labels to integers for confusion matrix.

print(f"Validation labels: {validation_labels}", len(validation_labels))  # Print the validation labels and their count. 