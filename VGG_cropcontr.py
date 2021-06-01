import os
import cv2
import re
import glob
import numpy as np
import random
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from sklearn.model_selection import train_test_split
#import histocolor
import tensorflow
import bkg_pix
from tensorflow.python.keras import Sequential

#from keras.models import Sequential
#from keras.layers import InputLayer
#from keras.models import Sequential, Model
#from keras.layers.core import Flatten, Dense, Dropout, Lambda, Reshape
#from keras.layers import Input
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.layers import Conv2D, MaxPooling2D, Activation
#from keras.optimizers import SGD, RMSprop, Adam


l2 = tensorflow.keras.regularizers.l2
example_model = tensorflow.keras.Sequential()
BatchNormalization = tensorflow.keras.layers.BatchNormalization
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Activation = tensorflow.keras.layers.Activation
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense
#Sequential = tensorflow.python.keras.Sequential
#InputLayer = keras.layers.InputLayer

callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

Regularizer = l2(0.0001)
print("Loaded all libraries")

fpath = "/home/mfusco/singlegrain"
file = open("output400.txt", "w")
alpha=3
beta=0
random_seed = 88
# find and enumerates classes
categories = os.listdir(fpath)
categories = categories[:2]
print("List of categories = ", categories, "\n\nNo. of categories = ", len(categories))


def load_images_and_labels(categories):
    img_lst = []
    labels = []
    classlabel = []
    d = []
    h = []
    i = 0
    for index, category in enumerate(categories):
        for i in range(30000):
            random_img = random.choice(os.listdir(fpath + "/" + category))
            img1 = cv2.imread(fpath + "/" + category + "/" + random_img)
            #img = Image.open(fpath + "/" + category + "/" + random_img)
            if img1 is None:
                break
            #conv = bkg_pix.nobkg(img, img1)
            #conv = no_noise.bkg_remove(img)

            adjusted = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)
            img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            cropped_image = img[30:66, 30:66]
            img_array = Image.fromarray(cropped_image, 'RGB')
            #if img_array.size != (96, 96):
             #   break
       	    #grainid = re.findall("g(\d+).png", random_img)
            # = re.findall("v(\d+)", random_img)
            #d.append(grainid[0])
            #h.append(viewid[0])
            img_lst.append(np.array(img_array))
            labels.append(index)

    return img_lst, labels


images, labels = load_images_and_labels(categories)
print("No. of images loaded = ", len(images), "\nNo. of labels loaded = ", len(labels))
print(type(images), type(labels))

images = np.array(images)
labels = np.array(labels)

print("Images shape = ", images.shape, "\nLabels shape = ", labels.shape)
print(type(images), type(labels))

# 1-step in data shuffling

# get equally spaced numbers in a given range
n = np.arange(images.shape[0])
print("'n' values before shuffling = ", n)
# shuffle all the equally spaced values in list 'n'
np.random.seed(random_seed)
np.random.shuffle(n)
print("\n'n' values after shuffling = ", n)

# 2-step in data shuffling

# shuffle images and corresponding labels data in both the lists
images = images[n]
labels = labels[n]
print("Images shape after shuffling = ", images.shape, "\nLabels shape after shuffling = ", labels.shape)

# data normalization
images = images.astype(np.float32)
labels = labels.astype(np.int32)
images = images / 255
print("Images shape after normalization = ", images.shape)

# Split the loaded dataset into train, test sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=random_seed)

print("x_train shape = ", x_train.shape)
print("y_train shape = ", y_train.shape)
print("\nx_test shape = ", x_test.shape)
print("y_test shape = ", y_test.shape)

model = Sequential()

model.add(Conv2D(input_shape=(36,36,3),filters=24,kernel_size=(3,3),padding="same", activation="relu"))
#model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=48,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=48,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=2, activation="softmax"))

#opt = "adam"(lr=0.001)
model.summary()

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# train the model
H = model.fit(x_train, y_train, epochs=100,  learning_rate=0.0001, batch_size=16, callbacks=[callback], validation_data=(x_test, y_test), shuffle=True)
# metrics
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

pred = model.predict(x_test)
pred.shape
print(pred)
np.savetxt(file, pred, delimiter=",", newline="\n")
file.close()

accuracy = H.history['acc']
val_accuracy = H.history['val_acc']
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', color="red", label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('vgg_acc.png')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b', color="red", label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('vgg_loss.png')
plt.show()

# number of images correctly and incorrectly classified by the model
correct = np.where(pred == y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:4]):
    plt.subplot(2, 2, i + 1)
    plt.imshow(x_test[correct], cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(pred[correct], y_test[correct]))
    plt.tight_layout()
plt.show()

incorrect = np.where(pred != y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
for j, incorrect in enumerate(incorrect[:4]):
    plt.subplot(2, 2, j + 1)
    plt.imshow(x_test[incorrect], cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(pred[incorrect], y_test[incorrect]))
    plt.tight_layout()
plt.show()

# classification report
from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(2)]
print(classification_report(y_test, pred, target_names=target_names))

# Display few random images with actual vs predicted values of labels
plt.figure(1, figsize=(19, 10))
n = 0
