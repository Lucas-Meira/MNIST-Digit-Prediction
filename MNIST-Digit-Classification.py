import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

class_names = []
for i in range(10):
    class_names.append(i)

print('Training data:', train_images.shape, train_labels.shape)
print('Test data:', test_images.shape, test_labels.shape)

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

    # Normalize data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32')

train_images = train_images / 255.0
test_images = test_images / 255.0
'''
model = tf.keras.models.Sequential() 

model.add(tf.keras.layers.Flatten(input_shape=(28,28),name = 'flatten')) # Add the first layer. Flatten is not a layer of neurons but it flatens the image tensor [28,28] into a pixel vector [784] 
model.add(tf.keras.layers.Dense(256,activation='relu',name='dense-128-relu')) # Add a dense layer. 128 respects the input size >= # neurons >= size of output
model.add(tf.keras.layers.Dense(10,activation='softmax',name='dense-10-softmax')) # Output layer with 10 neurons (10 images)
'''

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

print('Input Shape: ', train_images.shape)
print()
print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

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

predictions = model.predict(test_images)

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

count = 0
error_index = []
for i in range(int(len(test_images))):
    if(test_labels[i] != np.argmax(predictions[i])):
        count+=1
        error_index.append(i)

count

model.save("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\Digit_Recognition_Models.h5")