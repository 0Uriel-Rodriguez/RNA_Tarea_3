#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers


# In[1]:



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
model.save("red3c3.h5")
exit()


# In[8]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam

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
                    epochs=5,
                    verbose=1,
                    validation_data=(x_testv, y_testc))

score = model.evaluate(x_testv, y_testc, verbose=1) 
print(score)
a=model.predict(x_testv) 
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])
model.save("red3c3.h5")
exit()


# In[9]:


#-------------------------------------------------Regularización L1---------------------------------------------------------
modelo = tf.keras.models.load_model('red3c3.h5')
modelo = Sequential()
modelo.add(Dense(1000, activation='sigmoid',kernel_regularizer=regularizers.L1(0.001) ,input_shape=(784,)))
modelo.add(Dense(500, activation='sigmoid',kernel_regularizer=regularizers.L1(0.01)))
modelo.add(Dense(300, activation='sigmoid',kernel_regularizer=regularizers.L1(0.001)))
modelo.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.L1(0.02)))
modelo.summary()
modelo.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_testv, y_testc))
exit()


# In[10]:


#---------------------------------------Regularización L2---------------------------------------------------------------

modelo = tf.keras.models.load_model('red3c3.h5')
modelo = Sequential()
modelo.add(Dense(1000, activation='sigmoid', input_shape=(784,)))
modelo.add(Dense(500, activation='sigmoid',kernel_regularizer=regularizers.L2(l2=1e-4)))
modelo.add(Dense(300, activation='sigmoid',kernel_regularizer=regularizers.L2(l2=1e-4)))
modelo.add(Dense(num_classes, activation='softmax'))
modelo.summary()
modelo.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_testv, y_testc))
exit()


# In[14]:


#---------------------------------------Regularización L1L2---------------------------------------------------------------

modelo = tf.keras.models.load_model('red3c3.h5')
modelo = Sequential()
modelo.add(Dense(1000, activation='sigmoid', input_shape=(784,)))
modelo.add(Dense(500, activation='sigmoid',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
modelo.add(Dense(300, activation='sigmoid',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
modelo.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
modelo.summary()
modelo.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_testv, y_testc))
exit()


# In[15]:


#---------------------------------------Regularización L1L2 y Dropout-----------------------------------------------------
modelo = tf.keras.models.load_model('red3c3.h5')
modelo = Sequential()
modelo.add(Dense(1000, activation='sigmoid', input_shape=(784,)))
modelo.add(Dropout(0.2))
modelo.add(Dense(500, activation='sigmoid',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
modelo.add(Dropout(0.1))
modelo.add(Dense(300, activation='sigmoid',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
modelo.add(Dropout(0.5))
modelo.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
modelo.summary()
modelo.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=100,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_testv, y_testc))
exit()


# In[ ]:




