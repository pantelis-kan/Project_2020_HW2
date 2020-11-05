import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 1


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import adam,RMSprop
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import UpSampling2D,Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def read_input(filename, num_images):

    f = open(filename, "rb")
    image_size = 28

    #ignore header
    f.read(16)

    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255
    data = data.reshape(num_images, image_size, image_size,1)
    return data

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(28, 28, 1))(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small & thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small & thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded


def show_image(img):
    image = np.asarray(img).squeeze()
    plt.imshow(image, cmap='gray')
    #plt.imshow(image)
    plt.show()


input_filename = "train-images-idx3-ubyte"
test_filename = "train-images-idx3-ubyte"

input_num_images = 60000
test_num_images = 10000

train_data = read_input(input_filename,input_num_images)
test_data = read_input(test_filename,test_num_images)

print("Original image:\n")
show_image(train_data[1])
print(train_data.shape)
print(train_data.dtype)

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)


# Use the initial data as a training set and validation set. 
# 80% of the data will be the training set
# 20% of the data will be the validation set
train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,train_data, 
                                                        test_size=0.2, random_state=13)

batch_size = 128
epochs = 1

input_img = keras.Input(shape=(28,28,1))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.gpu_device_name())

autoencoder = keras.Model(input_img, decoder(encoder(input_img))) 
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

#autoencoder.summary()

# Start the fitting process 
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
                                epochs=epochs,verbose=1,
                                validation_data=(valid_X, valid_ground))

# Visualize train and validation loss
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()                                    