#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import time
import imutils
from imutils.video import VideoStream
import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# In[2]:


data_paths = list(paths.list_images('dataset')) # This sis the directory of the dataset. Since it is thesame folder
                                                # with my script, so i  need only the name
full_data =[] # Used to collect the images
labels = [] # used to collect the labels
for image_path in data_paths:
    label = image_path.split(os.path.sep)[-2] # get the label of each image from the name of the image.
    image = img_to_array(load_img(image_path, target_size=(224, 224)))
    image = preprocess_input(image) #used a tensorflow package to preprocess the image in a specific format
                                    #to enable it work with mobilenetv2 model.
    
    full_data.append(image) #collects the image
    labels.append(label) # Collects the labels
data =np.array(full_data,dtype='float32') # converts image to numpy 
labels= np.array(labels)  # converts labels to numpy


# In[3]:


# Converts the labels of the image to categorical data
lb = LabelBinarizer()
raw_label = labels
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
labels


# In[4]:


len(full_data)


# In[5]:


masked = []
for image_path in list(paths.list_images('dataset\with_mask'))[:5]: # Loads only five image from the directory
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converts to RGB images
    masked.append(image)
    
    # Display images with Mathplotlib
fig, axes = plt.subplots(1, 5, figsize=(20, 20))
for img,label,ax in zip(masked[:5],raw_label[2165:2170],axes):
    ax.set_title([label])
    ax.imshow(img)
    ax.axis('off')
plt.show()


# In[6]:


unmasked = []
for image_path in list(paths.list_images('dataset\without_mask'))[:5]: # Loads only five image from the directory
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converts to RGB images
    unmasked.append(image)
    
    # Display images with Mathplotlib
fig, axes = plt.subplots(1, 5, figsize=(20, 20))
for img,label,ax in zip(unmasked[:5],raw_label[:5],axes):
    ax.set_title([label])
    ax.imshow(img)
    ax.axis('off')
plt.show()


# In[7]:


(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=250)


# In[8]:


Main_model = MobileNetV2(weights="imagenet", include_top=False,  # The pre-made mobilenetv2 model from tensorflow keras
                         input_tensor=Input(shape=(224, 224, 3)))

Main_out_put = Main_model.output

# We now create our own output layer to use with the main model
head_model = AveragePooling2D(pool_size = (2,2))(Main_out_put)
Flatten_layer = Flatten(name="flatten")(head_model)
Dense_layer = Dense(128, activation="relu")(Flatten_layer)
Dropouts = Dropout(0.5)(Dense_layer)
#Dense_layer = Dense(64, activation="relu")(Flatten_layer)
#Dropouts = Dropout(0.5)(Dense_layer)
Output_layer = Dense(2, activation="softmax")(Dropouts) # Output layer of the head.


model = Model(inputs = Main_model.input, outputs = Output_layer) # Joining the main model and the created output layer.

for layer in Main_model.layers: #freezing the layers of the main model so that they would not be updated in the first run.
    layer.trainable = False


# In[9]:


class Optimizer:
    def __init__(self, model, mb = 8, lr = 0.0001, loss = tf.keras.losses.binary_crossentropy, #initialization function
               opt=tf.keras.optimizers.Adam, regularization = "l1",lamda = 0.01):
        self.model     = model # model all the way from main class model 
        self.loss      = loss #tf.keras.losses.MeanSquaredError()
        self.optimizer = opt(learning_rate = lr) # passing the learning rate to the uptimization function.
        self.mb        = mb  #minni batch size
        self.l1_l2_regul = self.regularization_type(regularization) #selects specific regularization type
        self.reg_const = lamda #regularization parameter

        #Train and test losses and accuracy
        self.train_loss     = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()

        self.test_loss     = tf.keras.metrics.Mean()
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy()

  
    @tf.function #Training function
    def train_step(self, x , y):
        with tf.GradientTape() as tape:
            predictions = model(x) # Make predictions
            loss_temp = self.loss(y, predictions) #Compare the prediction with the ground truth to get the loss.
            loss = self.apply_reg(loss_temp) # Apply regularization

        #Applying gradients
        gradients = tape.gradient(loss, self.model.trainable_variables) # Get the trainable weights.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # apply back propergation.
        self.train_loss(loss)  # Update the train loss
        self.train_accuracy(y, predictions) # Get train accuracy
        return loss

    @tf.function # This is the test step
    def test_step(self, x , y):
        predictions = self.model(x) # Make predictions
        loss = self.loss(y, predictions) #Compare the prediction with the ground truth to get the loss.
        self.test_loss(loss) # Update the test loss
        self.test_accuracy(y, predictions) # Get updatad test accuracy
  
    def apply_reg(self,dyn_loss): # This function applies regularization to the weights during training
        reg_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES) #gets the weights
        regularizer = self.l1_l2_regul(self.reg_const)(reg_loss) #using tf.keras.regularizers
        loss = tf.reduce_mean(dyn_loss + regularizer)
        return loss

  #function to recognize and implement the needed regularization according to the imputed argument
    def regularization_type(self,querry):
        if querry == "l1":
            return regularizers.l1
        elif querry == "l2":
            return regularizers.l2

  # Trains the model by mini batches mb
    def train (self):
        batches =0
        for mbX, mbY in self.Augumentation.flow(self.trainX,self.trainY, batch_size = self.mb):
            self.train_step(mbX, mbY)
            batches +=1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
  
  #tests the model also in batches
    def test  (self):
        batches = 0
        for mbX, mbY in self.Augumentation.flow(self.testX,self.testY, batch_size = self.mb):
            self.test_step(mbX, mbY)
            batches +=1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
  
  # the run function
    def run   (self, dataX, dataY, testX, testY, epochs, verbose=2):
        historyTR = [] # collects the training loss history
        historyTS = [] # collects the test loss histopry
        historyTR_acc = [] # collects the training Accuracy history
        historyTS_acc = [] # collects the training Accuracy history
        template = '{} {}, {}: {}, {}: {}, {}: {},{}: {}' # for displaying outputs during training
                                                
        ###### Data Augumentation ######
        # This is done authomatically during the training process using the keras function ImageDataGenerator.
        # At each epoch, it splits the image into batches and generates variation of sample images for each individual image.
        # The type of samples to generate is imputed inside the function
        #######################################
        self.Augumentation = ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,
                                   height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
        self.Augumentation.fit(x_train)

        # Collects the imput data and pass them into a global variable of the class
        self.trainX = dataX
        self.trainY = dataY 
        self.testX  = testX
        self.testY = testY 

        #training loop
        for i in range(epochs):
      
            self.train () # calls train step
            self.test  () # calls test step immidiately after train step
              #prints train and test loss and accuracy for display
            if verbose > 0:
                print(template.format("epoch: ", i+1," TRAIN LOSS: ", self.train_loss.result(),
                              " TEST LOSS: " , self.test_loss.result(),
                              " TRAIN ACC: " , self.train_accuracy.result()*100,
                              " TEST ACC: "  , self.test_accuracy.result()*100))
      
              #gathers training information data for visualization
            temp = '{}'
            historyTR.append(float(temp.format(self.train_loss.result())))
            historyTS.append(float(temp.format(self.test_loss.result() )))
            historyTR_acc.append(float(temp.format(self.train_accuracy.result()*100)))
            historyTS_acc.append(float(temp.format(self.test_accuracy.result()*100 )))
              #resets the loss and accuracy history after each epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
        
        return historyTR, historyTS,historyTR_acc,historyTS_acc


# In[10]:


#models  = model()
# Call the optimization function with imput parameters
opt    = Optimizer (model, mb = 20, lr = 0.0001,regularization = "l1",lamda = 0.01 ) 

# call the run function inside the optimization class with the datasets
los_t,los_v,acc_t,acc_v = opt.run (x_train, y_train, x_test, y_test, 10, verbose=1)


# In[11]:


model.summary()


# In[12]:


# Display the details of the training process
N = 10
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), los_t, label="train_loss")
plt.plot(np.arange(0, N), los_v, label="val_loss")
plt.plot(np.arange(0, N), acc_t, label="train_acc")
plt.plot(np.arange(0, N), acc_v, label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
#.savefig(args["plot"])


# In[13]:


# Model accuracy
correct = 0
total = 0
pred = np.argmax(model.predict(x_test), axis=1)

for i, img in enumerate(pred):
    if img == np.argmax(y_test[i]):
        correct += 1
    total += 1

print(correct/total * 100)


# In[14]:


# Classification report
cr = classification_report(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis = 1))
print(cr)


# In[15]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
Y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="OrRd",linecolor="black", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[16]:


# Saving the model
model.save('my_model.h5')


# In[ ]:




