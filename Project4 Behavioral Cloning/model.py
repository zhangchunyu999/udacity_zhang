import csv
import cv2
import numpy as np
import os
import sklearn
import random
from sklearn.utils import shuffle
import matplotlib.image as mpimg

samples=[]

with open('data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples  = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
                
            images=[]
            measurements=[]
                
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path=batch_sample[i]
                    filename=source_path.split('/')[-1] 
                    images_path='data/IMG/'+filename
                    img=mpimg.imread(images_path)
                        
                    images.append(img)
                        
                    if i==0:
                        measurement=float(batch_sample[3])
                    if i==1:
                        measurement=float(batch_sample[3])+0.2
                    if i==2:
                        measurement=float(batch_sample[3])-0.2
                    
        
                    measurements.append(measurement)       
            
            augmented_images, augmented_measurements=[], []
            for img, measurement in zip(images,measurements):
                augmented_images.append(img)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(img,1))
                augmented_measurements.append(measurement*-1.0)
                    
            X_train=np.array(augmented_images)
            y_train=np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
       
batch_size=32

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Convolution2D(64,3,3,activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(.3))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

model.save('model.h5')
exit()