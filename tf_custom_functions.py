# Libraries
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from google.colab import drive
import zipfile
import pathlib
import datetime as dt

############################################################
# Visualization Functions
############################################################

def show_random_images(image_set, label_set, class_labels):
  """
  Shows 10 random images with class labels from the dataset.
  Dtype of the set must be a NumPy array.
  Observations of image_set and label_set must be in same order.

  Args:
    image_set: Image dataset to be randomly chosen to plot.
    label_set: Numerical labels related to image_set to represent class_labels.
    class_labels: List of class labels, to be referred by label_set.
  
  Returns:
    None
  """
  # Create random indexes
  random_idx_list = random.sample(range(len(image_set)), 10)

  # Show images
  for index, idx in enumerate(random_idx_list):
    plt.subplot(2, 5, index+1)
    plt.imshow(image_set[idx], cmap = 'gray')
    plt.title(class_labels[label_set[idx]])
    plt.axis(False);


def plot_label_frequencies(dataset_classes, class_labels):
  """
  Plots frequencies of class labels.

  Args:
    dataset_classes: Class label list which contains
                     numerical representations of classes.
    class_labels: Class label list which contains text representations
                  of classes.
  
  Returns:
    None
  """
  # Create DataFrame for labels
  class_labels_df = pd.DataFrame({
      'count' : pd.Series(dataset_classes).value_counts().sort_index(),
      'label' : class_labels
  })

  # Plot frequencies
  plt.figure(figsize = (8, 6))
  sns.barplot(data = class_labels_df, x = 'label', y = 'count')

  plt.title("Frequencies of Classes")
  plt.xlabel('Labels')
  plt.ylabel('Frequency')
  plt.show()


def plot_accuracy_and_loss(history):
  """
  Plots accuracy and loss graphs seperately for a fitted model.

  Args:
    history: History dictionary of the model.
  """

  # Calculate the epochs
  epochs = range(1, len(history.history['accuracy']) + 1)

  # Plot a figure
  plt.figure(figsize = (14, 6))

  # Plot the accuracy graph
  plt.subplot(1, 2, 1)
  plt.title("Train and Validation Accuracy")
  plt.plot(epochs, history.history['accuracy'], label = "Train Accuracy")
  plt.plot(epochs, history.history['val_accuracy'], label = "Validation Accuracy")
  plt.legend()

  # Plot the loss graph
  plt.subplot(1, 2, 2)
  plt.title("Train and Validation Loss")
  plt.plot(epochs, history.history['loss'], label = "Train Loss")
  plt.plot(epochs, history.history['val_loss'], label = "Validation Loss")
  plt.legend()

############################################################
# Colab & Kaggle
############################################################

def connect_kaggle():
  """
  Connects to Kaggle via API key located in Google Drive.
  """
  # Mount Drive 
  drive.mount('/content/drive')
  # Set Kaggle configuration
  os.environ['KAGGLE_CONFIG_DIR'] = '/content/drive/MyDrive/kaggle'

############################################################
# Data Preparation
############################################################

def extract_data(zip_file : str,
                 data_path : str = '/content/data/'):
  """
  Extracts zipfile into data path given.

  Args:
    zip_file (str): Path of zip folder contains data.
    data_path (str): Path data folder that data to be extracted.
  """
  # Create data folder if does not exist
  data_path_ = pathlib.Path(data_path)
  if data_path_.is_file():
    print(f"{data_path_} folder already exists.")
  else:
    print(f"{data_path_} folder does not exist, creating new one...")
    os.mkdir(data_path_)
  
    # Extract zipfile into data folder
    print("Extracting the zip folder...")
    zip_ref = zipfile.ZipFile(zip_file)
    zip_ref.extractall(path = data_path_)
    zip_ref.close()
    print("Zip folder has extracted.")


def create_train_valid_test_df(train_path,
                               validation_path,
                               test_path):
  """
  Creates train, validation and test datasets from folder paths.

  Args:
    train_path (Path): Path of train folder.
    validation_path (Path): Path of validation folder.
    test_path (Path): Path of test folder.

  Returns:
    (train_df, validation_df, test_df)
  """
  # Create train dataframe
  train_image_paths = [str(pathlib.Path(path)) for path in list(train_path.glob('*/*'))]
  train_image_labels = [path.parent.name for path in list(train_path.glob('*/*'))]

  train_df = pd.DataFrame({
      'image' : train_image_paths,
      'label' : train_image_labels
  })

  # Create validation dataframe
  validation_image_paths = [str(pathlib.Path(path)) for path in list(validation_path.glob('*/*'))]
  validation_image_labels = [path.parent.name for path in list(validation_path.glob('*/*'))]

  validation_df = pd.DataFrame({
      'image' : validation_image_paths,
      'label' : validation_image_labels
  })

  # Create test dataframe
  test_image_paths = [str(pathlib.Path(path)) for path in list(test_path.glob('*/*'))]
  test_image_labels = [path.parent.name for path in list(test_path.glob('*/*'))]

  test_df = pd.DataFrame({
      'image' : test_image_paths,
      'label' : test_image_labels
  })

  # Print information about folders
  print(f"There are {len(train_df)} images in train folder.")
  print(f"There are {len(validation_df)} images in validation folder.")
  print(f"There are {len(test_df)} images in test folder.")

  return train_df, validation_df, test_df

############################################################
# Callbacks
############################################################

def create_tensorboard_callback(dir_name,
                                experiment_name):
  """
  Creates TensorBoard callback to save model experimentation results
  into related folder.
  """
  log_dir = dir_name + '/' + experiment_name + dt.datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
  print(f"Saving TensorBoard log files to {log_dir}")
  return tensorboard_callback

############################################################
# Model Creation
############################################################

def create_model(model_url,
                 num_classes,
                 image_shape):
  """
  Creates Keras Sequential from model URL.

  Args:
    model_url (str): URL of the model taken from Kaggle.
    num_classes (int): Number of classes to be classified.
    image_shape (Tuple): Input shape of images given in tuple format.
  
  Returns:
    An uncompiled Keras Sequential model with model_url as feature extractor
    layer and Dense output layer with num_classes output neurons.
  """
  # Download the pre-trained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable = False,
                                           name = "feature_extractor_layer",
                                           input_shape = image_shape)
  
  # Create our own model
  model = tf.keras.Sequential([
      feature_extractor_layer,
      tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'output_layer')
  ])

  # Return the model
  return model