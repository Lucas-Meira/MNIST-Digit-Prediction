import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import time

np.random.seed(int(time.time()))
# Class that reads images from dataset user-created dataset with the dataset creation tool
class dataset():
  def __init__(self):
    # The files are locates at ./datasets
    self.file = "datasets"
  
  #Method for loading data. It reads num_images files
  def load_data(self, num_images):
    
    #Separe test and train images. Custom is set to 85% train images and 10% test images
    self.images = [i for i in range(int(num_images*0.85))]
    self.labels = [i for i in range(int(num_images*0.85))]
    self.test_images = [i for i in range(int(num_images*0.15))]
    self.test_labels = [i for i in range(int(num_images*0.15))]
    i = 0
    j = 0

    # Get random indexes
    random_indexes = [i for i in range(1,101)]
    np.random.shuffle(random_indexes)
    
    for m in range(10):
      for k in range(num_images//10):
        print("Reading data from ." + os.sep + self.file + os.sep +"{}".format(m)+ os.sep +"{}.npy".format(random_indexes[(k+1)%100]))
        if(k < num_images*0.85/10):
          self.images[i] = np.load(self.file+ os.sep +"{}".format(m)+ os.sep +"{}.npy".format(random_indexes[k]))
          self.labels[i] = m
          i += 1
        else:
          self.test_images[j] = np.load(self.file+ os.sep +"{}".format(m)+ os.sep +"{}.npy".format(random_indexes[k]))
          self.test_labels[j] = m
          j +=1

    return (np.array(self.images), np.array(self.labels)), (np.array(self.test_images), np.array(self.test_labels))

data = dataset() # Create an object of dataset

#Load data
(train_images, train_labels), (test_images, test_labels) =  data.load_data(1000) #tf.keras.datasets.mnist.load_data()

# Uncomment if you want to train with MNIST Dataset
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Create labels for the digits, 0 - 9
class_names = []
for i in range(10):
    class_names.append(i)

print('Training data:', train_images.shape, train_labels.shape)
print('Test data:', test_images.shape, test_labels.shape)

# Functions for displaying training and test images
def show_training_image(index):
    img_label = str(train_labels[index])
    plt.figure()
    plt.title('Image Label ' + img_label)
    plt.imshow(train_images[index], cmap = 'gray') # data is grayscale, but displays in color without cmap='gray'
    plt.colorbar()
    plt.show()

def show_test_image(index):
    img_label = str(test_labels[index]) + ' (' + class_names[int(test_labels[index])] + ')'
    plt.figure()
    plt.title('Image Label ' + img_label)
    plt.imshow(test_images[index], cmap = 'gray') # data is grayscale, but displays in color without cmap='gray'
    plt.colorbar()
    plt.show()

# Reshape images into (1,28,28,1) format
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32')

# Mnist dataset contains images with pixel values from 0 to 255
# Normalize data to a range from 0 to 1 if it is not already normalized
if(np.max(train_images) > 250):
  train_images = train_images / 255.0
  test_images = test_images / 255.0

# Uncomment if you want to create a whole new dataset and not train a hybrid NN
'''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
'''
model = tf.keras.models.load_model(os.path.join("Models","Digit_Recognition_Models.h5"))

print('Input Shape: ', train_images.shape)
print()
print(model.summary())

# Compile the model with ADAM optimizer and Sparse Categorical Crossentropy loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Tensorboard configuration
log_dir = os.path.join("logs","fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Early stopping calback for stopping when test images loss stops decreasing after 4 epochs (Patience)
#early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

# Train model
train_hist = model.fit(x=train_images, 
          y=train_labels, 
          epochs=5, 
          validation_data=(test_images, test_labels), 
          callbacks=[tensorboard])

def plot_acc(hist):
  # plot the accuracy
  plt.title('Accuracy History')
  plt.plot(hist.history['accuracy'])
  plt.ylabel('Accuracy')
  plt.xlabel('epoch')
  plt.show(block=False)
  
def plot_loss(hist):
  # plot the loss
  plt.title('Loss History')
  plt.plot(hist.history['loss'])
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.show(block=False)

plot_loss(train_hist)
plot_acc(train_hist)

# Predict all test images
predictions = model.predict(test_images)

# Plot a test image with its prediction
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.reshape((28, 28)).astype('float32'), cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# Create a bar plot with the probabilities of being each digit

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

import time
np.random.seed(int(time.time()))

pred_img_index = np.random.randint(0,len(test_images)-1,size=num_images)
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(pred_img_index[i], predictions[pred_img_index[i]], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(pred_img_index[i], predictions[pred_img_index[i]], test_labels)
plt.tight_layout()
plt.show()

# Count the number of images the model has mistakenly predicted and get their indexes
count = 0
error_index = []
for i in range(int(len(test_images))):
    if(test_labels[i] != np.argmax(predictions[i])):
        count+=1
        error_index.append(i)

count

model.save(os.path.join("Models","Dataset_and_MNIST_Digit_Recognition_Models_{}.h5".format(datetime.datetime.now().strftime("%H%M"))))