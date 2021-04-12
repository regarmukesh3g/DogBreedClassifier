#!/usr/bin/python

import argparse
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

from PIL import Image
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def get_parser():
    """
    Creates and return CLI parser.
    :return:
    """
    parser = argparse.ArgumentParser(description='Trains the data model')
    parser.add_argument('image_path', metavar='image file path', type=str,
                        default='/home/rm3g/Desktop/Shoppee/priceMatching/test_images/0006c8e5462ae52167402bac1c2e916e.jpg',help = 'path of the image file')
    return parser




# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


from tensorflow.keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

from tensorflow.keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))



from extract_bottleneck_features import *
Resnet50_model = Sequential()
Resnet50_model.add(Conv2D(1024,1,input_shape=(7,7,2048)))
Resnet50_model.add(Dense(512, activation='relu'))
Resnet50_model.add(GlobalAveragePooling2D())
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Resnet50_model.load_weights('saved_models/weights.best.Resnet50.hdf5')

saved_dict = torch.load('dog_names.pth')
dog_names = saved_dict['dog_names']

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    print(bottleneck_feature.shape)
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def predict_breed(file_path):
    if face_detector(file_path) or dog_detector(file_path):
        pred_breed = Resnet50_predict_breed(file_path)
    else:
        print("Error: Given image does not contain any Dog.")
    return pred_breed

def main():
    parser = get_parser()
    args = parser.parse_args()
    img_path = args.image_path
    pred_breed = predict_breed(img_path).split('.')[-1]
    img = Image.open(img_path)
    print(pred_breed)
    plt.title(pred_breed)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
