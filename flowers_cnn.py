import os

from matplotlib import pyplot as plt
import tensorflow as tf
import random
import numpy as np
import platform
from tensorflow.keras import layers, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
# classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

class FlowersCNN:
    def __init__(self):
        pass

    def prepare_data(self, raw_data_dir):        
        # 1 First turn the folder contents into a list of files:
        data = os.listdir(raw_data_dir)
        for data_folder in data:
            # 2 Then check if the file is a directory:
            if os.path.isdir(os.path.join(raw_data_dir, data_folder)):
                # 3 If it is a directory, then we can assume it is a class folder:
                print(f"Found class folder: {data_folder}")
                # If there is at least one class folder, we can assume that the data is structured correctly.
                class_folder = os.path.join(raw_data_dir, data_folder)
                class_files = os.listdir(class_folder)
                if os.path.exists(os.path.join(raw_data_dir, 'train', data_folder)):
                    break  # Skip if the train folder already exists
                if os.path.exists(os.path.join(raw_data_dir, 'validation', data_folder)):
                    break
                # Create a train folder for the class if it does not exist
                os.makedirs(os.path.join(raw_data_dir, 'train', data_folder))
                os.makedirs(os.path.join(raw_data_dir, 'validation', data_folder))
                sample_size = len(class_files) * 0.2
                validation_samples = random.sample(range(len(class_files)), int(sample_size)) # Randomly select 20% of the files for validation. 
                # The variable holds the indices of the files to be used for validation.
                print(len(validation_samples), f"out of {len(class_files)} files will be used for validation.", f"Ratio of validation files is {round(sample_size / len(class_files), 2) * 100}%")     
                for j in range(len(class_files)):
                    # if the index of the file is in the validation_samples list, then we copy it to the validation folder.
                    # Otherwise, we copy it to the train folder.
                    if j in validation_samples:
                        # Copy the 20% of the files to the validation folder
                        os.system(f"cp {os.path.join(class_folder, class_files[j])} {os.path.join(raw_data_dir, 'validation', data_folder)}")
                    else:
                        # Copy the remaining 80% of the files to the train folder
                        os.system(f"cp {os.path.join(class_folder, class_files[j])} {os.path.join(raw_data_dir, 'train', data_folder)}")
                print(f"Ratio of train files is: {len(os.listdir(os.path.join(raw_data_dir, 'train', data_folder))) / len(class_files)}")
            else:
                # 4 If it is not a directory, then we can assume it is an image file:
                print(f"Found image file: {data_folder}. Not doing anything with it.")
        print("Data preparation complete.")

    def preprocess_images(self, data_dir, shuffle=True, augment_data=True):
        # 1 You have folders with images in the data_dir, which will be either train or validation.
        # You now want to preprocess the images in these folders.
        
        # First, you need to create a dataset from the images:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(180, 180),
            crop_to_aspect_ratio=True,
            shuffle=shuffle,
            seed=42
        )
        self.classes = train_ds.class_names  # Get the class names from the dataset.
        # train_ds is a tf.data.Dataset object that contains the images and their labels.
        # This is like a generator that produces 4D - 1D pairs of images and labels.
        # 2 Now you can preprocess the images:
        # 2.1 You can normalize the images to the range [0, 1] by dividing the pixel values by 255.0:
        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label
        train_ds = train_ds.map(normalize)
        # 2.2 You can also apply data augmentation to the images:
        data_augmentation_crop = tf.keras.Sequential([
            # ex: Across 25 epochs, that single original image will be seen 25 times, but in 25 different augmented forms
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),  # Randomly rotate the image by a factor of 0.10, that's up to 10% of 360 degrees, or 36 degrees.
            layers.RandomContrast(0.1),
            layers.RandomTranslation(0.1, 0.1), 
            # RandomCrop + Resizing: Imagine cutting out a smaller rectangle from the photo, then enlarging it to fill the frame — the subject looks bigger, and fine details may blur slightly.
            layers.RandomCrop(120, 120),  # Randomly crop the image to 120x120 pixels. Retains about 44% of the original image area.
            layers.Resizing(180, 180),  # Resize back to the original size of 180x180 pixels.
        ])
        data_augmentation_translate = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10), # Randomly rotate the image by a factor of 0.10, that's up to 10% of 360 degrees, or 36 degrees.
            
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomZoom(0.2),
            # Randomly translate the image by a factor of 0.1 (10%) both horizontally and vertically,
            # small shifts (e.g., 0.1, 0.1) help the model handle off-center subjects.
            # RandomTranslation: Imagine sliding a printed photo under a fixed‑size frame — you see a different part of the same photo, but the subject’s size doesn’t change.
            layers.RandomContrast(0.2),
        ])
        data_augmentation_basic = tf.keras.Sequential([
            # ex: Across 25 epochs, that single original image will be seen 25 times, but in 25 different augmented forms
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),  # Randomly rotate the image by a factor of 0.15, that's up to 15% of 360 degrees, or 54 degrees.
            layers.RandomZoom(0.2), # Randomly zoom in/out by a factor of 0.2 (20%)
            layers.RandomContrast(0.2),
            
        ])
        def augment(image, label):
            if tf.random.uniform(()) > 0.5:
                image = data_augmentation_crop(image)
            # elif tf.random.uniform(()) > 0.5 and tf.random.uniform(()) <= 0.9:
            #     image = data_augmentation_translate(image)
            else:
                image = data_augmentation_translate(image)
            return image, label
        if augment_data == True:
            train_ds = train_ds.map(augment)
        # 2.3 Finally, you can ignore errors in the dataset, such as corrupted images:
        train_ds = train_ds.apply(tf.data.Dataset.ignore_errors)
        # 3 Now you can return the preprocessed dataset:
        return train_ds
    
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=(180, 180, 3)),  # Input layer for images of size 180x180 with 3 color channels
            layers.Conv2D(16, (3, 3), activation='relu'),
            
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            # Rule of thumb: Dropout early in the network disrupts low-level feature learning (risky),
            # dropout late in the network disrupts high-level memorization (usually good). 
            # The smaller your dataset, the more you benefit from late dropout; the larger your dataset, the less you may need it at all.
            layers.Dropout(0.2),  # Dropout layer to reduce overfitting
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.classes), activation='softmax')  # Output layer for multi-class classification
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model
    
    def visualise_training(self, history_object, show=True):
        """
        Visualises accuracy and loss during training.
        """
        plt.figure(figsize=(8, 8))  # Start fresh figure
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
        if show:
            plt.show()
        else:
            plt.savefig('./stats/training_history.png')
        plt.close()  # Clear the figure for future plots.

    def train_model(self, train_ds, validation_ds, epochs=3, class_weights=None):
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "best_flowers_model.keras",   # filename
            save_best_only=True,          # only keep the best epoch
            monitor="val_loss",           # which metric to watch
            mode="min"                    # lower is better
        )

        history = self.build_model().fit(
            train_ds,
            validation_data=validation_ds,
            epochs=epochs,
            callbacks=[checkpoint_cb], # At the end of training, 
            # the file on disk isn’t necessarily the weights from the final epoch — it’s the weights from the epoch where your chosen metric (in this case, lowest val_loss) was best.
            class_weight=class_weights
        )
        plot_training = input("Do you want to plot the training history? (yes/no): ").lower() in ['yes', 'y'] # returns True if the user inputs 'yes' or 'y', otherwise False
        
        if plot_training:
            show_plot = input("Do you want to see the plot? If not, it will simply be saved. (yes/no): ").lower() in ['yes', 'y'] # returns True if the user inputs 'show' or 's', otherwise False
            self.visualise_training(history, show=show_plot)
        # save_model = input("Do you want to save the model? (yes/no): ").lower() in ['yes', 'y'] # returns True if the user inputs 'yes' or 'y', otherwise False
        # if save_model:
        #     model_name = input("What name do you want to save the model as?: ")
        #     self.model.save(f'./saved_models/{model_name}.keras')
        #     print(f"Model saved as '{model_name}.keras'.")
        return history
    
    def flowers_predict(self, model, image_path):
        # Preprocess the image to match the input shape of the model
        loaded_model = tf.keras.models.load_model(model)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(180, 180)) # a Pillow image object — basically a Python-friendly wrapper around the raw pixel data.
            #So the flow is usually:
                # PIL image → convert to NumPy array → normalize → feed into TensorFlow/PyTorch/etc.
        image = np.array(image)  # Convert the image to a NumPy array
        image = image / 255.0  # Normalize the image to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        # Make prediction
        predictions = loaded_model.predict(image) # Produces an array of probabilities for each class. 
        # The array has 2 dimensions: the first dimension is the batch size (1 in this case), and the second dimension is the number of classes. The output is a 2D array with shape (1, num_classes).
        predicted_class = np.argmax(predictions, axis=1) # Get the index of the class with the highest probability. 
        # Axis 1 means we are looking for the maximum value along the columns. The rows are the different images in the batch.
        # predicted_class is a 1D array with the index of the predicted class.
        print(self.classes)  # Print the class names for debugging.
        predicted_class_name = self.classes[predicted_class[0]]  # Get the class name from the index
        confidence = np.max(predictions)  # Get the confidence of the prediction
        confidence_perc = float(confidence) * 100  # Convert to percentage
        confidence = f"{confidence_perc:.2f}%"  
        print(f"Predicted class: {predicted_class_name} with confidence: {confidence}")   
        return predicted_class_name, confidence_perc # The data type of the returned object is a tuple containing the predicted class name and the confidence score.


if platform.system() == 'Windows':
    raw_data_dir = r'C:\python_work\tensorFlow\wsl_venv\Udacity\flowers\flower_photos'
else:
    raw_data_dir = '/mnt/c/python_work/tensorFlow/wsl_venv/Udacity/flowers/flower_photos'

flowers_cnn = FlowersCNN()
flowers_prepare_data = flowers_cnn.prepare_data(raw_data_dir=raw_data_dir)  # This will create the train and validation folders in the flower_photos directory.
flowers_preprocessed_train_ds = flowers_cnn.preprocess_images(os.path.join(raw_data_dir, 'train'), augment_data=True, shuffle=True) # This will return a tf.data.Dataset object that contains the preprocessed images and their labels.

"""
You normally do not shuffle the validation (or test) set.
Validation/Test sets are supposed to be a stable, fixed benchmark. If you shuffle them every time, you risk making debugging harder (metrics vary slightly run to run, even with the same model).
Training set benefits from shuffling because it prevents the model from memorizing the order of examples and improves generalization.
Frameworks like image_dataset_from_directory default to shuffle=True, but that option is mostly useful for training. For validation/test, you’d typically set shuffle=False to preserve order.
"""
flowers_preprocessed_validation_ds = flowers_cnn.preprocess_images(os.path.join(raw_data_dir, 'validation'), augment_data=False, shuffle=False)

if __name__ == "__main__":
    trained_flowwers = flowers_cnn.train_model(flowers_preprocessed_train_ds, flowers_preprocessed_validation_ds, epochs=60)
    print("Model training complete.")
# flowers_model.save('flowers_cnn_model.h5')
# print("Model saved as 'flowers_cnn_model.h5'.")
# You can now use this model to classify flower images.
# To load the model, use: tf.keras.models.load_model('flowers_cnn_model.h5').

