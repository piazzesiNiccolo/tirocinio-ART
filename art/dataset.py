import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob


def set_label(root):
    for obj in root.findall('object'):
        value = obj.find('name').text
        if value == "vehicle":
            return np.array([0, 1])
    return np.array([1, 0])


def load_carla_train_dataset():
    
    train_path = '../dataset/train'
    
    x_train = []
    
    for file in glob.glob('/'.join([train_path,"annotations","*.xml"])):
        root = ET.parse(file).getroot()
        name = root.find('filename')
        i = '/'.join([train_path,"images",name.text])
        img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        x_train.append([np.array(img),set_label(root)])
    return x_train

def load_carla_test_dataset():
    
    test_path = '../dataset/test'
    
    x_test = []
    
    for file in glob.glob('/'.join([test_path,"annotations","*.xml"])):
        root = ET.parse(file).getroot()
        name = root.find('filename')
        i = '/'.join([test_path,"images",name.text])
        img = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        x_test.append([np.array(img),set_label(root)])
    return x_test

def load_carla_dataset():
    return load_carla_train_dataset(), load_carla_test_dataset()


if __name__ == '__main__':
    load_carla_dataset()
