# -*- coding: utf-8 -*-
"""autoencoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Dmxhy0tFNAbDRCvrzIl_G4zAA5nED6eM

Import all the necessary libraries
"""

#from google.colab import drive
#drive.mount("/content/drive", force_remount=True)

import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam,RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import UpSampling2D,Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

"""Define 3 functions: 
1.   **read_input** : reads a binary file consisting of images, and returns a numpy array with shape 
(num_images,128,128,1)
2.   **encoder** : constructs the encoded version of the image. Default : 4 layers
3. **decoder** : follows the reverse steps from the encoder. Default : 4 layers
4. **show_image** : prints an image with matplotlib
5. **fully_connected** : Create a fully connected layer
"""

def read_labels(filename, num_images):
    f = open(filename, "rb")
    f.read(8)
    buf = f.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def read_input(filename, num_images):

    f = open(filename, "rb")
    image_size = 28

    #ignore header
    f.read(16)

    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255
    data = data.reshape(num_images, image_size, image_size,1)
    return data


layer_size = 32
enlayers = 0
pooling_layers_pos = []
total_layers = 0

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    global enlayers
    global layer_size 
    global pooling_layers_pos
    global total_layers

    layer_size = 32
    pooling_layers_pos.clear()

    temp = input("Insert number of layers for your encoder: ")
    enlayers = int(temp)
    total_layers = enlayers

    for l in range(enlayers):
        if (l == 0):
            choice = input("For first layer you can only choose Conv. If you agree press 1. Other option will terminate the programme: ")
            if (int(choice) != 1): sys.exit("Terminating the programe...")
            x = Conv2D(layer_size, (3, 3), activation='relu', padding='same',input_shape=(28, 28, 1))(input_img) #28 x 28 x 32
            x = BatchNormalization()(x)
            total_layers = total_layers + 1 
            continue;
        else:
            choice = input("If you want Conv layer press 1, if you want maxPooling press 2: ")
      
        choice = int(choice)
      
        if (choice == 1): #conv layer with BN
            layer_size = layer_size * 2
            x = Conv2D(layer_size, (3, 3), activation='relu', padding='same')(x) #same x same x 64/128/256/etc
            x = BatchNormalization()(x)
            total_layers = total_layers + 1 
        elif (choice == 2):
            x = MaxPooling2D(pool_size=(2, 2))(x) # double x double x same
            pooling_layers_pos.append(l)
            if (l == enlayers - 1): sys.exit("Your last layer on encoder cannot be a maxpooling one. Programme will terminate now...")
        else:
            sys.exit("Wrong choice! Programme will terminate now...")

    return x
    

def decoder(x):
    #decoder
    global pooling_layers_pos    #we need to know when we have a maxpooling
    global enlayers
    global layer_size


    for l in range(enlayers):
        if (l == enlayers - 1 ):
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # 28 x 28 x 1
             
        
        if l in pooling_layers_pos:
            x = UpSampling2D((2,2))(x) # half x half x same
        else:
            x = Conv2D(layer_size, (3, 3), activation='relu', padding='same')(x) #same x same x half
            x = BatchNormalization()(x)
            layer_size = layer_size / 2

    return decoded


def show_image(img):
    image = np.asarray(img).squeeze()
    plt.imshow(image, cmap='gray')
    #plt.imshow(image)
    plt.show()

# Create a fully connected layer
def fully_connected(enco,drop_param):
    flat = Flatten()(enco)
    #drop1 = Dropout(drop_param)(flat)
    den = Dense(128, activation='relu')(flat)
    drop2 = Dropout(drop_param)(den)
    # Since we have 10 classes, output a filter with size 10
    out = Dense(10, activation='softmax')(drop2)
    return out

"""Read the binary files for test,train and label data.

Normalize data in the [0,1] scale

Seperate the input into train and validation set

Default : 80% train data , 20% validation data
"""

from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

input_filename = "/content/drive/My Drive/ML/data/train-images-idx3-ubyte"
#test_filename = "/content/drive/My Drive/ML/data/t10k-images-idx3-ubyte"

#input_label_filename = "/content/drive/My Drive/ML/data/train-labels.idx1-ubyte"
#test_label_filename = "/content/drive/My Drive/ML/data/t10k-labels.idx1-ubyte"


if (sys.argv[1] == '-d'):
    print(sys.argv[1])
else:
    sys.exit("Wrong Parameters...")


input_num_images = 60000
test_num_images = 10000

train_data = read_input(input_filename, input_num_images)
test_data = read_input(test_filename, test_num_images)

train_labels = read_labels(input_label_filename, input_num_images)
test_labels = read_labels(test_label_filename, test_num_images)


#print("Original image:\n")
#show_image(train_data[1])
print(train_data.shape)
print(train_data.dtype)

#Convert labels from categorical representation to one-hot encoding
#train_Y_one_hot = to_categorical(train_labels)
#test_Y_one_hot = to_categorical(test_labels)


# Normalize data in the 0-1 scale
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

# Use the initial data as a training set and validation set. 
# 80% of the data will be the training set
# 20% of the data will be the validation set
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,train_data, 
                                                        test_size=0.2, random_state=13)

"""Create a keras model and copile with mean_squared_error loss function and RMSprop optimizer.

Start the fitting process with 50 epochs and 128 batch size as default.

Save the model with .h5 extension and plot the training and validation loss
"""

repeat = True
model_path = '/content/drive/My Drive/ML/data/results/'
#        model_path = input("Insert the path to save the models: ")

while (repeat == True):
    batch_size = input("Batch size: ")
    epochs = input("Epochs: ")

    batch_size = int(batch_size)
    epochs = int(epochs)

    input_img = keras.Input(shape=(28,28,1))

    autoencoder = keras.Model(input_img, decoder(encoder(input_img))) 
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

    #autoencoder.summary()

#    global enlayers

    # Start the fitting process 
    autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
                                    epochs=epochs,verbose=1,
                                    validation_data=(valid_X, valid_ground))

    h5name = "layers" + str(enlayers) + "_epochs"+ str(epochs) +"_batch"+ str(batch_size)

    user_option = input("If you want to see the graphic result of the losses press 1: ")
    if (int(user_option) == 1):
        # Visualize train and validation loss
        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        epochs = range(epochs)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        #plt.savefig('/content/drive/My Drive/ML/data/results/' + h5name + '.png')
        plt.show()

    user_option = input("If you want to save the model, press 1")
    if (int(user_option) == 1):
        autoencoder.save(model_path  + h5name + '.h5')

    user_option = input("If you want to repeat the experiment press 1. Any other option will terminate.")
    if (int(user_option) == 1):
        repeat = True
    else:
        repeat = False

"""Select a model and load the .h5 file."""

modelfile = "/content/drive/My Drive/ML/data/results/layers4_epochs50_batch64.h5"

load_model = keras.models.load_model(modelfile)

load_model.summary()

"""Start the prediction process for the test set (decode the 10,000 images from the already compiled model)

Plot a subset of the original and the reconstructed images for comparison purposes.
"""

pred = load_model.predict(test_data)

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_data[i, ..., 0], cmap='gray')
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')  
plt.show()