#!/usr/bin/env python
# coding: utf-8



from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
from torchvision import models
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms, models
# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets





import cv2                
import matplotlib.pyplot as plt                        


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# ### (IMPLEMENTATION) Assess the Human Face Detector
# 
# __Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
# - What percentage of the first 100 images in `human_files` have a detected human face?  
# - What percentage of the first 100 images in `dog_files` have a detected human face? 
# 
# Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.
# 
# __Answer:__ 

# In[5]:


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.

human_results = []
for img_path in human_files_short:
    face_exists = face_detector(img_path)
    human_results.append(face_exists)

precision = np.array(human_results).sum()/ len(human_results)
print("Precision: {}".format(precision))


dog_results = []
for img_path in dog_files_short:
    face_exists = face_detector(img_path)
    dog_results.append(face_exists)

recall = 1 - (np.array(dog_results).sum()/ len(dog_results))
print("Recall: {}".format(recall))



from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 


# human_results = []
# for img_path in human_files_short:
#     face_exists = dog_detector(img_path)
#     human_results.append(face_exists)
#
# precision = 1 - (np.array(human_results).sum()/ len(human_results))
# print("Recall: {}".format(precision))
#
#
# dog_results = []
# for img_path in dog_files_short:
#     face_exists = dog_detector(img_path)
#     dog_results.append(face_exists)
#
# recall = np.array(dog_results).sum()/ len(dog_results)
# print("Precision: {}".format(recall))



from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']


# ### (IMPLEMENTATION) Model Architecture
# 
# Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
#     
#         <your model's name>.summary()
#    
# __Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.
# 
# __Answer:__ We have seen that Resnet50 was able to successfully differentiate between human and dogs. I am thinking that it will be better at classifying all other images in dog breeds as well. I am adding some activation functions as well. 
# 
# 

# In[26]:


train_Resnet50.shape


# In[27]:


### TODO: Define your architecture.
Resnet50_model = Sequential()
Resnet50_model.add(Conv2D(1024,1,input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(512, activation='relu'))
Resnet50_model.add(GlobalAveragePooling2D())
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()


# ### (IMPLEMENTATION) Compile the Model

# In[28]:


### TODO: Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# ### (IMPLEMENTATION) Train the Model
# 
# Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  
# 
# You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 

# In[29]:


### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def predict_breed(file_path):
    
    if face_detector(file_path) or dog_detector(file_path):
        pred_breed = Resnet50_predict_breed(file_path)
    else:
        print("Error: Given image does not contain any Dog.")
    return pred_breed
    

human_files_short = human_files[300:305]
dog_files_short = train_files[300:305]
from torchvision import transforms
transform = transforms.Resize(224,224)
print("Predicting on human images")
i = 1
plt.figure(figsize=[20,30])
for img_path in human_files_short:
    img = Image.open(img_path)
    plt.subplot(2,5,i)
    i += 1
    pred_breed = predict_breed(img_path).split('.')[-1]
    print(pred_breed)
    plt.title(pred_breed)
    plt.imshow(img)
    
    
print("Predicting on Dog images") 
for img_path in dog_files_short:
    img = Image.open(img_path)
    plt.subplot(2,5,i)
    i += 1
    pred_breed = predict_breed(img_path).split('.')[-1]
    print(pred_breed)
    plt.title(pred_breed)
    plt.imshow(img)
    #plt.show()
plt.show()


# In[ ]:




