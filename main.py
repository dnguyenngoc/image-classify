import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array, smart_resize
import numpy as np
from elasticsearch import Elasticsearch

# Load VGG16
model = VGG16()
model.summary()

# Load Elasticsearch
es = Elasticsearch(['host': '10.1.32.130', 'port': '9200'])

image_path= r"./datasets/identity_card/mat_truoc_1.png"


img = load_img(image_path)

ORIGIN_IMAGE_SIZE = (224,224)


def get_feature_map_encode(img):
    input = img_to_array(img) 
    input = smart_resize(input, ORIGIN_IMAGE_SIZE, interpolation='bilinear')
    input = np.expand_dims(input, axis=0)
    feature_maps = model.predict(input)
    feature_maps_list = feature_maps[0].tolist()
    # tensor = tf.convert_to_tensor(feature_maps)
    return feature_maps_list


# def upload_dataset_on_elasticseach(datasets_dir):