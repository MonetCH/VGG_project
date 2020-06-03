import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping
#-------------------------------------------------------------
#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)  
#-------------------------------------------------------------
#variables
num_classes = 3
#-------------------------------------------------------------
#path=train data的位置
path_pos="C://MER_Challenge/dataset/training/SAMM_CASME_pos"
path_neg="C://MER_Challenge/dataset/training/SAMM_CASME_neg"
path_sur="C://MER_Challenge/dataset/training/SAMM_CASME_sur"
#設定空陣列
x_train, y_train, x_test, y_test = [], [], [], []
for i in os.listdir(path_pos):
    if "006" in i:
        path1=os.path.join(path_pos,i)
        for j in os.listdir(path1):
            img = cv2.imread(os.path.join(path1,j),0)
            img=np.array(img).astype('float32')
            x_train.append(img)
            y_train.append(keras.utils.to_categorical(0, num_classes))
for i in os.listdir(path_neg):
    if "006" in i:
        path1=os.path.join(path_neg,i)
        for j in os.listdir(path1):
            img = cv2.imread(os.path.join(path1,j),0)
            img=np.array(img).astype('float32')
            x_train.append(img)
            y_train.append(keras.utils.to_categorical(1, num_classes))
for i in os.listdir(path_sur):
    if "006" in i:
        path1=os.path.join(path_sur,i)
        for j in os.listdir(path1):
            img = cv2.imread(os.path.join(path1,j),0)
            img=np.array(img).astype('float32')
            x_train.append(img)
            y_train.append(keras.utils.to_categorical(2, num_classes))
#-------------------------------------------------------------
#將矩陣中的數值float化
x_train = np.array(x_train,'float32')
y_train = np.array(y_train,'float32')
x_test = np.array(x_test,'float32')
y_test = np.array(y_test,'float32')

x_train, x_test, y_train, y_test=train_test_split(x_train,y_train,test_size=0.2)
#-------------------------------------------------------------
x_train = x_train.reshape(x_train.shape[0], 128, 128, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 128, 128, 1).astype('float32')
#-------------------------------------------------------------
#x_train、y_train、x_test、y_test的numpy的陣列大小
print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)

#train model
model=Sequential()
model.add(Conv2D(96, (11, 11),strides=(4,4),activation='relu', input_shape=(128,128,1)))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(256, (5, 5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(384, (3, 3),padding='same',activation='relu'))
model.add(Conv2D(384, (3, 3),padding='same',activation='relu'))
model.add(Conv2D(256, (3, 3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
#-------------------------------------------------------------
#batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train,batch_size=32)
#-------------------------------------------------------------
#model evaluate
Adam=keras.optimizers.Adam(lr=0.00001, beta_1=0.9
                           , beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy'
              , optimizer=Adam
              , metrics=['accuracy']
              )
#-------------------------------------------------------------
filepath = "C://MER_Challenge/model/AlexNet.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1)
checkpoint2 = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
callbacks_list = [checkpoint1,checkpoint2]
epochs = 30
history=model.fit_generator(train_generator,validation_data=(x_test,y_test)
                            ,verbose=1,steps_per_epoch=len(x_train)/16
                            ,epochs=epochs,callbacks = callbacks_list)
#-------------------------------------------------------------
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc','val_acc'], loc='upper left')
plt.show()
#-------------------------------------------------------------
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc='upper left')
plt.show()