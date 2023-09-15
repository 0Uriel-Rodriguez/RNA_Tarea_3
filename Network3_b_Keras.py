#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

#---------------------------------------------------Experimento 2------------------------------------------------------
# Solo aunemto de neuronas en la primera capa densa 
# Cambio de Optimizador a Adam
# Solo capa de activacion sigmoid
dataset=mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')
x_trainv /= 255
x_testv /= 255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(500, activation='sigmoid', input_shape=(784,)))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_testv, y_testc))

score = model.evaluate(x_testv, y_testc, verbose=1) 
print(score)
a=model.predict(x_testv) 
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])


# In[3]:


model.save("red3b1.h5")


# In[14]:


#---------------------------------------------------Experimento 1------------------------------------------------------
# Solo aunemto de neuronas y de capas densas  
# Optimizador RMSprop 
# Capa de activacion sigmoid
dataset=mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')
x_trainv /= 255
x_testv /= 255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(200, activation='sigmoid', input_shape=(784,)))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=30,
                    verbose=1,
                    validation_data=(x_testv, y_testc))

score = model.evaluate(x_testv, y_testc, verbose=1) 
print(score)
a=model.predict(x_testv) 
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])


# In[12]:


#---------------------------------------------------Experimento 3------------------------------------------------------
# Solo aunemto de neuronas y de capas densas  
# Optimizador Adam 
# Capa de activacion sigmoid y la ultima softmax 
dataset=mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')
x_trainv /= 255
x_testv /= 255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_shape=(784,)))
model.add(Dense(500,activation='sigmoid'))
model.add(Dense(300,activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=40,
                    verbose=1,
                    validation_data=(x_testv, y_testc))

score = model.evaluate(x_testv, y_testc, verbose=1) 
print(score)
a=model.predict(x_testv) 
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])


# In[ ]:




