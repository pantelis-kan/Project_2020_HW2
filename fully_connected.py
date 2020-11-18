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


def read_labels(filename, num_images):
    f = open(filename, "rb")
    f.read(4)

    num_images = int.from_bytes(f.read(4), 'big')
#    print("Total images in labels file: ", num_images)

    buf = f.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def read_input(filename, num_images):

    f = open(filename, "rb")
    image_size = 28

    #ignore header
    f.read(4)

    num_images = int.from_bytes(f.read(4), 'big')
#    print("Total images in file: ", num_images)

    f.read(8)

    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255
    data = data.reshape(num_images, image_size, image_size,1)
    return data

def show_image(img):
    image = np.asarray(img).squeeze()
    plt.imshow(image, cmap='gray')
    #plt.imshow(image)
    plt.show()


global activation_method

# Create a fully connected layer
def fully_connected(enco,drop_param):
    flat = Flatten()(enco)
    drop1 = Dropout(drop_param)(flat)
    den = Dense(128, activation='relu')(drop1)
    drop2 = Dropout(drop_param)(den)
    # Since we have 10 classes, output a filter with size 10
    activation_option = input("Choose an activation method (1 for softmax, 2 for linear, 3 for sigmoid): ")
    
    if int(activation_option) == 1 :
      activation_method = "softmax"
    elif int(activation_option) == 2:
      activation_method = "linear"
    else:
      activation_method = "sigmoid"  

    out = Dense(10, activation=activation_method)(drop2)  
    return out      


# Handle command line arguments
if len(sys.argv) < 10:
    exit("Please give ALL the necessairy arguments. Exiting...")

args = 0

if "-d" in sys.argv:
    input_filename = sys.argv[sys.argv.index("-d") + 1]
    args += 1
if "-t" in sys.argv: 
    test_filename = sys.argv[sys.argv.index("-t") + 1]
    args += 1
if "-dl" in sys.argv:
    input_label_filename = sys.argv[sys.argv.index("-dl") + 1]
    args += 1
if "-tl" in sys.argv: 
    test_label_filename = sys.argv[sys.argv.index("-tl") + 1]
    args += 1     
if "-model" in sys.argv: 
    modelfile = sys.argv[sys.argv.index("-model") + 1]
    args += 1 

print(args)
if args != 5:
    exit("Please give ALL the necessairy arguments. Exiting...")

print(input_filename)
print(input_label_filename)
print(test_filename)
print(test_label_filename)



input_num_images = 60000
test_num_images = 10000

train_data = read_input(input_filename,input_num_images)
test_data = read_input(test_filename,test_num_images)

train_labels = read_labels(input_label_filename,input_num_images)
test_labels = read_labels(test_label_filename, test_num_images)


#print("Original image:\n")
#show_image(train_data[1])
print(train_data.shape)
print(train_data.dtype)

#Convert labels from categorical representation to one-hot encoding
train_Y_one_hot = to_categorical(train_labels)
test_Y_one_hot = to_categorical(test_labels)

#print(train_Y_one_hot[0])

# Normalize data in the 0-1 scale
train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)



#modelfile = "/content/drive/My Drive/ML/data/results/layers6_epochs50_batch128.h5"

load_model = keras.models.load_model(modelfile)
total_layers = (len(load_model.layers) / 2) + 1 
print(total_layers)
layer_names=[layer.name for layer in load_model.layers]




repeat = True
while (repeat == True):
  percent = -1
  while percent > 100 or percent < 0:
    percent = input("Choose test data % (0-100): ")
    percent = int(percent)

  train_X,valid_X,train_label,valid_label = train_test_split(train_data,train_Y_one_hot,test_size=int(percent)/100,random_state=13)

  drop_param = input("Choose dropout parameter: ")

  optimizer_option = input("Choose optimizer (1 for Adam, everything else for RMSPprop)")

  drop_param = float(drop_param)
  optimizer_option = int(optimizer_option)

  outputs = [layer.output for layer in load_model.layers]
  encoded_output = outputs[int(total_layers)-1]

  input_img = keras.Input(shape=(28,28,1))

  full_model = keras.Model(outputs[0],fully_connected(encoded_output,drop_param))


  # Use the weights from the encoder part of the autoencoder to create the new model
  for l1,l2 in zip(full_model.layers[:int(total_layers)],load_model.layers[0:int(total_layers)]):
      l1.set_weights(l2.get_weights())


  # Do NOT train the encoder part again. It was already done in the previous steps
  for layer in full_model.layers[0:int(total_layers)]:
    layer.trainable = False


  if optimizer_option == 1:
    optimizer_method = "Adam"
    # Compile the new model with cross entropy since we want classification 
    full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
  else:
    optimizer_method = "RMSprop"
    # Compile the new model with cross entropy since we want classification 
    full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])


  #full_model.summary()

  fc_epochs = input("Choose epochs: ")
  fc_batch_size = input("Choose batch size: ")
  fc_epochs = int(fc_epochs)
  fc_batch_size = int(fc_batch_size)

  classify_train = full_model.fit(train_X, train_label, batch_size=fc_batch_size,epochs=fc_epochs,verbose=1,validation_data=(valid_X, valid_label))

  full_model_name = "testSplit_"+str(percent)+"_2dropout_FullyConnected_epochs"+str(fc_epochs)+"_batch"+str(fc_batch_size)+"_dropout"+str(drop_param) +optimizer_method+".h5"
  #full_model.save('/content/drive/My Drive/ML/data/fully_connected_results/'+full_model_name)
  full_model.save(full_model_name)


  accuracy = classify_train.history['accuracy']
  val_accuracy = classify_train.history['val_accuracy']
  loss = classify_train.history['loss']
  val_loss = classify_train.history['val_loss']
  epochs = range(len(accuracy))
  plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
  plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  #plt.savefig('/content/drive/My Drive/ML/data/fully_connected_diagrams/' + full_model_name + '.png')
  plt.savefig(full_model_name + '.png')
  plt.show()

  test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)

  print('Test loss:', test_eval[0])
  print('Test accuracy:', test_eval[1])

  user_option = input("If you want to repeat the experiment press 1. Any other option will terminate: ")
  if (int(user_option) == 1):
      repeat = True
  else:
      repeat = False

#fully_connected_modelfile = "/content/drive/My Drive/ML/data/fully_connected_results/new_2dropout_FullyConnected_epochs70_batch512_dropout0.58Adam.h5"
fully_connected_modelfile = full_model_name
full_model_load = keras.models.load_model(fully_connected_modelfile)


predicted_classes = full_model_load.predict(test_data)

# Round the predicted values to the closest integer
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==test_labels)[0]
print("Found %d correct labels" % len(correct) )

for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes!=test_labels)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.tight_layout()



from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(10)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))        