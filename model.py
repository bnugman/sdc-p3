import csv
import cv2
import keras
import numpy as np


steers = []
images = []


for subdir in ["track1-take2", "track1-take3"]:
    with open("./sim_data/%s/driving_log.csv" % subdir) as f:
        reader = csv.reader(f)
        for line in reader:
            steer = float(line[3])

            # when there's actual steering, use the center image
            # otherwise, use images from side cameras, with corrections

            if False: #np.abs(steer) < 0.001:
                index_and_correction = {0: 0.0}
            else:
                index_and_correction = {1: 0.15, 2: -0.15}

            for index, correction in index_and_correction.items():
                imgfile = line[index].split("/")[-1]
                curpath = "./sim_data/%s/IMG/%s" % (subdir, imgfile)
                img = cv2.imread(curpath)
                images.append(img)
                steers.append(steer + correction)

                # also add mirror images
                # images.append(cv2.flip(img, 1))
                # steers.append(- (steer + correction))


X_train = np.array(images)
y_train = np.array(steers)

print(X_train.shape)
print(steers[:100])

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# crop
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# convolutional and pooling layers
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# now, Dense with Drop-outs, for high expressive power with good generalization
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)

model.save('model.h5')
