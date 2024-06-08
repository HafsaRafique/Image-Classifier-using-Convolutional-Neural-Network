import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
import numpy as np
import cv2 as cv      #pip install opencv-python
from tensorflow import keras 
from keras import datasets, models, layers


import matplotlib.pyplot as plt
# dataset is already in keras
(training_images, training_labels),(test_images, test_labels)= datasets.cifar10.load_data() #images are arrays of pixels
#normalize the pixels so values are b/w 0 and 1
training_images= training_images/255
test_images= test_images/255

#assign names to numbers
names_classes= ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
for i in range(16):
    plt.subplot(4,4, i+1)
    #remove coordinate system
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap='binary') #need images in binary
    plt.xlabel(names_classes[training_labels[i][0]])
plt.show()    

training_images= training_images[:20000]
training_labels= training_labels[:20000]
test_images= test_images[:4000]
test_labels=test_labels[:4000]
                                  ###########   may comment out from here############
model= models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3) )) #neurons, convolution matrix
model.add(layers.MaxPooling2D((2,2))) #simplifies essential info
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten()) 

#Convolution extracts important features from image, Maxpool extracts essential info from image

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) #for probabilities

model.compile(optimizer='adam',loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(training_images, training_labels))

loss, accuracy= model.evaluate(test_images, test_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")


model.save('image_classifier.keras')


                                   ##################till here (after saving the model) ###########################
                                   #####   SAVE THE MODEL AND COMMENT OUT ABOVE PORTION BEFORE CONTINUING BELOW  ##########
model= models.load_model('image_classifier.keras')

img= cv.imread('deer.jpg')
img= cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap='binary')

resized_img = cv.resize(img, (32, 32))

# Reshape the image to match the model's input shape and add a batch dimension
img_array = np.reshape(resized_img, (1, 32, 32, 3))

# Normalize the pixel values
img_array = img_array / 255.0

# Perform prediction
prediction = model.predict(img_array)
index= np.argmax(prediction) #activation of highest neuron
print(f"Prediction: {names_classes[index]}")
plt.show()