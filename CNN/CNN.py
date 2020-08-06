# Convolutional Neural Network

# Importing the libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
import numpy as np
from keras.preprocessing import image

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),  # make training much faster
        batch_size=32,  # how many images in each batch, 32 is a classic default
        class_mode='binary')  # cat, dog: binary

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()  # an object as sequence of layers

# Step 1 - Convolution, use add method to add layer
cnn.add(tf.keras.layers.Conv2D(filters=32,
                              kernel_size=3,
                              activation='relu',
                              input_shape=[64, 64, 3]))

# Step 2 - Pooling (Max Pooling)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,
                              kernel_size=3,
                              activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# # Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Part 4 - Making a single prediction
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
print("Array", test_image)
test_image = np.expand_dims(test_image, axis=0)
print("Vector", test_image)
result = cnn.predict(test_image)
print(training_set.class_indices)

# Encoding to specify which is 1 and which is 0
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print("Single Prediction: ", prediction)
