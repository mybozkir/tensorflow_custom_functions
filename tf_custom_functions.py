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
import tensorflow as tf
import shutil
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

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

def compare_histories(base_history,
                      fine_tune_history,
                      initial_epochs = 5):
  """
  Compares accuracy and loss values of the model for before and after
  fine tuning.
  """
  # Get base history accuracy and loss values
  base_train_acc = base_history.history['accuracy']
  base_train_loss = base_history.history['loss']

  base_val_acc = base_history.history['val_accuracy']
  base_val_loss = base_history.history['val_loss']

  # Get fine-tuned history aaccuracy and loss values
  fine_tune_train_acc = fine_tune_history.history['accuracy']
  fine_tune_train_loss = fine_tune_history.history['loss']

  fine_tune_val_acc = fine_tune_history.history['val_accuracy']
  fine_tune_val_loss = fine_tune_history.history['val_loss']

  # Combine accuracy and loss values
  total_train_acc = base_train_acc + fine_tune_train_acc
  total_val_acc = base_val_acc + fine_tune_val_acc

  total_train_loss = base_train_loss + fine_tune_train_loss
  total_val_loss = base_val_loss + fine_tune_val_loss

  # Create plots
  plt.figure(figsize = (8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_train_acc, label = 'Training Accuracy')
  plt.plot(total_val_acc, label = 'Validation Accuracy')
  plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label = 'Start Fine Tuning')
  plt.legend(loc = 'lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(total_train_loss, label = 'Training Loss')
  plt.plot(total_val_loss, label = 'Validation Loss')
  plt.plot([initial_epochs - 1, initial_epochs - 1], plt.ylim(), label = 'Start Fine Tuning')
  plt.legend(loc = 'lower right')
  plt.title('Training and Validation Loss');

# Let's create a function to display Confusion Matrix
def plot_confusion_matrix(y_true,
                          y_pred,
                          labels,
                          figsize = (30, 30)):
  """
  Plots confusion matrix for classification projects.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
    labels: Class label names.
    figsize: Figure size, defaults (30, 30).
  """
  ConfusionMatrixDisplay.from_predictions(y_true = y_true,
                                          y_pred = y_pred,
                                          display_labels = labels,
                                          cmap = 'Blues',
                                          xticks_rotation = 'vertical',
                                          ax = plt.figure(figsize = figsize, dpi = 200).subplots())

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
                 data_path : str = 'data/'):
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

def inspect_dir(path):
  for dirpath, dirnames, filenames in os.walk(path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'")

def create_train_test_df(train_path : str,
                         test_path : str,
                         pattern : str = '*/*'):
  """
  Creates train and test datasets from folder paths.

  Args:
    train_path (Path): Path of train folder.
    test_path (Path): Path of test folder.

  Returns:
    (train_df, validation_df, test_df)
  """
  # Create train dataframe
  train_image_paths = [str(Path(path)) for path in list(Path(train_path).glob(pattern))]
  train_image_labels = [path.parent.name for path in list(Path(train_path).glob(pattern))]

  train_df = pd.DataFrame({
      'image' : train_image_paths,
      'label' : train_image_labels
  })

  # Create test dataframe
  test_image_paths = [str(Path(path)) for path in list(Path(test_path).glob(pattern))]
  test_image_labels = [path.parent.name for path in list(Path(test_path).glob(pattern))]

  test_df = pd.DataFrame({
      'image' : test_image_paths,
      'label' : test_image_labels
  })

  # Print information about folders
  print(f"There are {len(train_df)} images in train folder.")
  print(f"There are {len(test_df)} images in test folder.")

  return train_df, test_df

def create_train_test_dirs(root_data_path : str,
                           new_data_path : str,
                           train_path : str,
                           test_path : str,
                           train_split_size : float = 0.8,
                           root_data_pattern: str = '*/*/*'):
  """

  Args:
    root_data_path (str): Root data folder contains raw dataset.
    new_data_path (str): New data folder to contains class folders with all
                         images (witout train, val, test seperation).
    train_path (str): Path for train directory to be created.
    test_path (str): Path for test directory to be created.
    train_split_size (float): Float value for percentage of train split size.
    root_data_pattern (str): Folder pattern for the usage of glob.
                             Ex: '*/*/*
  """
  # Create a list that contains all images
  total_image_list = []
  for image in Path(root_data_path).glob(root_data_pattern):
    total_image_list.append(image)

  # Copy all images into their class directories in new data path
  new_data_dir = Path(new_data_path)

  for idx, image in enumerate(total_image_list):
    if image.is_file():
      # Obtain class label
      class_name = os.path.basename(os.path.dirname(image))

      # Create target folder
      target_dir = new_data_dir / class_name
      target_dir.mkdir(parents = True, exist_ok = True)

      # Create uniqueness for all files
      unique_name = f"{image.parent.name}_{idx}_{image.name}"
      target_file = target_dir / unique_name

      # Copy files
      shutil.copy(src = image,
                  dst = target_file)

  # Create class_labels list
  class_labels = []
  for dirpath, dirnames, filenames in os.walk(new_data_dir):
    class_labels.append(dirnames)
  class_labels = class_labels[0]
  class_labels

  # Create train and test paths
  train_path = Path(train_path)
  test_path = Path(test_path)

  # Create train and test directories with class directories
  for path in [train_path, test_path]:
    path.mkdir(parents = True, exist_ok = True)

    for class_ in class_labels:
      class_path = path / class_
      class_path.mkdir(parents = True, exist_ok = True)

  # Copy files into class folders located in train and test directories
  for dir in new_data_dir.iterdir():
    image_list = list(dir.iterdir())
    train_list = image_list[ : int(len(image_list) * train_split_size)+1]
    test_list = image_list[int(len(image_list) * train_split_size)+1 : ]

    for image in train_list:
      shutil.copy(src = image,
                  dst = train_path / dir.stem)

    for image in test_list:
      shutil.copy(src = image,
                  dst = test_path / dir.stem)

  # Create train and test lists
  train_list = list(train_path.glob('*/*'))
  test_list = list(test_path.glob('*/*'))

  print(f"Train directory is created with {len(train_list)} in total.")
  print(f"Test directory is created with {len(test_list)} in total.")
  print(f"There are {len(train_list) + len(test_list)} images in total.")

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

# Firstful, let's create a function to create checkpoint callback.
def create_checkpoint(checkpoint_path : str):
  """
  Creates Keras checkpoint callback to return to the point when necessary.
  Saves best weights only, so model checkpoint path must be ended with
  '.weights.h5'. Also monitors validation accuracy for the best results.

  Args:
    checkpoint_path (str): Filepath for saving checkpoint file.
  
  Returns:
    ModelCheckpoint object.
  """
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath = checkpoint_path,
      save_weights_only = True,
      monitor = 'val_accuracy',
      save_best_only = True
  )

  return checkpoint_callback

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

############################################################
# Metrics
############################################################

def plot_metric(classification_report_dict,
                metric,
                class_names):
  """
  Plots any value among precision, recall, f1-score and support. One must give
  the metric name directly among [precision, recall, f1-score, support].
  Also returns sorted DataFrame for the metric type chosen with class labels.

  Args:
    classification_report_dict: Classification Report dictionary.
    metric: Metric type to be visualized.
    class_names = List of class names.
  
  Returns:
    DataFrame: DataFrame of the sorted values of given metric.
  """

  # Create metric score dictionary
  metric_scores = {}
  for key, value in classification_report_dict.items():
    if key == 'accuracy':
      break
    else:
      metric_scores[class_names[int(key)]] = value[metric]
  
  # Create metric DataFrame
  metric_df = pd.DataFrame({
      'class_name' : list(metric_scores.keys()),
      'metric' : list(metric_scores.values())
  }).sort_values(by = 'metric', ascending = False).reset_index().drop('index', axis = 1)

  plt.figure(figsize = (12, 6))
  sns.barplot(data = metric_df,
              x = 'class_name',
              y = 'metric')
  plt.title('F1-Score for Class Labels')
  plt.xlabel('Class Names')
  plt.ylabel(metric)
  plt.xticks(rotation = 90)
  plt.show()

  # Return the DataFrame
  return metric_df