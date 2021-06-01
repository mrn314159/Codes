import os
import cv2
import re
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.python.keras import Sequential
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
l2 = tensorflow.keras.regularizers.l2
example_model = tensorflow.keras.Sequential()
BatchNormalization = tensorflow.keras.layers.BatchNormalization
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Activation = tensorflow.keras.layers.Activation
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense
callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # per usare la prima GPU che ha ID 0
Regularizer = l2(0.000001)
print("Loaded all libraries")

fpath = "/home/mfusco/singlegrain"
file = open("output_vector2cl.txt", "w")
random_seed = 88
alpha=3 #contrast 1-3
beta=0    #brightness 0-100

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
            img = cv2.imread(fpath + "/" + category + "/" + random_img)
            if img is None:
                break
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            img = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
            cropped_image = img[30:66, 30:66]
            img_array = Image.fromarray(cropped_image, 'RGB')
            #if img_array.size != (96, 96):
            #    break
       	    grainid = re.findall("g(\d+).png", random_img)
            viewid = re.findall("v(\d+)", random_img)
            d.append(grainid[0])
            h.append(viewid[0])
            img_lst.append(np.array(img_array))
            labels.append(index)

    return img_lst, labels, d, h


images, labels, d, h = load_images_and_labels(categories)
print("No. of images loaded = ", len(images), "\nNo. of labels loaded = ", len(labels))
print(type(images), type(labels))

images = np.array(images)
labels = np.array(labels)

print("Images shape = ", images.shape, "\nLabels shape = ", labels.shape)
print(type(images), type(labels))

df = pd.DataFrame(columns=['Class Label', 'Grain_id', 'View_id'])
df['Class Label'] = labels
df['Grain_id'] = d
df['View_id'] = h
print(df)
df.to_csv('dataframe.csv', index=False)

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
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2,  random_state=random_seed)

print("x_train shape = ", x_train.shape)
print("y_train shape = ", y_train.shape)
print("\nx_test shape = ", x_test.shape)
print("y_test shape = ", y_test.shape)

# Define all layers in the ALEXNET CNN model--------------------------------------------------------
model = Sequential()
# 1 conv layer
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same", activation="relu", input_shape=(36, 36, 3), activity_regularizer=Regularizer, kernel_regularizer=Regularizer))

# 1 max pool layer
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

# 2 conv layer
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))

# 2 max pool layer
#model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#model.add(BatchNormalization())

# 3 conv layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))

# 4 conv layer
#model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))

# 5 conv layer
#model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))

# 3 max pool layer
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())

# 1 dense layer
model.add(Dense(128, input_shape=(36, 36, 3), activation="relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# 2 dense layer
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# 3 dense layer
model.add(Dense(2, activation="relu"))
#model.add(Dropout(0.5))
model.add(BatchNormalization())

# output layer
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# train the model
H = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, data_batches=32, learning_rate=0.000001 , shuffle=True)
# metrics
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

pred = model.predict(x_test)
preds = model.predict_classes(x_test)
pred.shape
print(pred)
print(preds)
np.savetxt(file, pred, delimiter=",", newline="\n")
file.close()

# pred = np.argmax(np.round(pred), axis=1)
print(pred.shape, y_test.shape)

accuracy = H.history['acc']
val_accuracy = H.history['val_acc']
loss = H.history['loss']
val_loss = H.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', color='red', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('contracc.png')
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b',  color='red', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('contrloss.png')
plt.show()


# number of images correctly and incorrectly classified by the model
correct = np.where(pred == y_test)[0]
print("Found %d correct labels" % len(correct))
#for i, correct in enumerate(correct[:4]):
#    plt.subplot(2, 2, i + 1)
#    plt.imshow(x_test[correct], cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(pred[correct], y_test[correct]))
#    plt.tight_layout()

#plt.savefig('correct_examples.png')
#plt.show()

incorrect = np.where(pred != y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
#for j, incorrect in enumerate(incorrect[:4]):
#    plt.subplot(2, 2, j + 1)
#    plt.imshow(x_test[incorrect], cmap='gray', interpolation='none')
#    plt.title("Predicted {}, Class {}".format(pred[incorrect], y_test[incorrect]))
#    plt.tight_layout()
#plt.savefig('incorrect_examples.png')
#plt.show()


from sklearn.metrics import classification_report
import seaborn as sns
target_names = ["Class {}".format(i) for i in range(2)]
print(classification_report(y_test, preds, target_names=target_names))
report = classification_report(y_test, preds, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('classification_report.csv', index=False)


print(confusion_matrix(y_test, preds))
plt.figure()
pollo=sns.heatmap(confusion_matrix(y_test, preds), annot = True)
fige = pollo.get_figure()
fige.savefig('conf_matrix.png', dpi=400)

