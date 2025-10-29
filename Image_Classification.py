
import zipfile
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
import numpy as np

import zipfile
with zipfile.ZipFile('C:\\Users\\Hazerra\\Downloads\\101_food_classes_10_percent.zip','r') as ref:
    ref.extractall()

import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class_names=os.listdir(r'E:\Git_ML\LANGCHAIN\Python\101_food_classes_10_percent\train')
total_class=len(os.listdir(r'E:\Git_ML\LANGCHAIN\Python\101_food_classes_10_percent\train'))
print(total_class)

def plot_random(classes,file_dir):
    plt.figure(figsize=(10,8))
    for i in range(1,7):
        plt.subplot(2,3,i)
        obj_name=random.choice(classes)
        path=os.path.join(file_dir,obj_name)
        class_names=os.listdir(os.path.join(file_dir,obj_name))
        rand_img=random.choice(class_names)
        img_path=os.path.join(path,rand_img)
        img=mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(obj_name)
        plt.axis('off')
plot_random(class_names,r'E:\Git_ML\LANGCHAIN\Python\101_food_classes_10_percent\train')


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen=ImageDataGenerator(rescale=1./255)
test_gen=ImageDataGenerator(rescale=1./255)

train_dir=r'E:\Git_ML\LANGCHAIN\Python\101_food_classes_10_percent\train'
test_dir=r'E:\Git_ML\LANGCHAIN\Python\101_food_classes_10_percent\test'

train_data=train_gen.flow_from_directory(directory=train_dir,batch_size=32,target_size=(224,224),seed=42)
test_data=test_gen.flow_from_directory(directory=test_dir,batch_size=32,target_size=(224,224),seed=42)

print(train_data)

model=Sequential([Conv2D(filters=10,kernel_size=3,activation='relu',input_shape=(224,224,3)),
                  Conv2D(10,3,activation='relu'),
                  BatchNormalization(),
                  MaxPool2D(pool_size=2,padding='valid'),
                  Conv2D(10,3,activation='relu'),
                  Conv2D(10,3,activation='relu'),
                  BatchNormalization(),
                  MaxPool2D(2),
                  Flatten(),
                  Dense(128, activation='relu'),
                  Dropout(0.4),
                  Dense(total_class,activation='softmax')])
model.compile(optimizer=tensorflow.keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_data,epochs=10,validation_data=test_data,validation_steps=10,steps_per_epoch=50)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Training accuracy')
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Test Loss')
plt.plot(history.history['val_accuracy'],label='Test accuracy')
plt.legend()
plt.show()

file=st.file_uploader('Upload your file')

def helper(img):
    
    img = tensorflow.io.decode_image(file.read(), channels=3)
    img=tensorflow.image.resize(img,size=[224,224])
    img=img/255.0
    img = tensorflow.expand_dims(img, axis=0)  # add batch dimension
    return img

if file is not None:
    image = helper(file)
    prediction = np.array(model.predict(image))
   
    idx=np.argmax(prediction)
    st.write('Prediction : ',class_names[idx])
