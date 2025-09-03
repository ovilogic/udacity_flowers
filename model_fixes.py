from flowers_cnn import flowers_cnn
from flowers_cnn import flowers_preprocessed_train_ds, flowers_preprocessed_validation_ds
from diagnostic_tools import class_weights

trained_again = flowers_cnn.train_model(flowers_preprocessed_train_ds, flowers_preprocessed_validation_ds, epochs=60, class_weights=class_weights)
