from elasticsearch import Elasticsearch
import glob
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array, smart_resize
import numpy as np

# Load Elasticsearch
es = Elasticsearch([{'host': '10.1.32.130', 'port': '9200'}])

# Load VGG16
model = VGG16()
model.summary()

def create_elasticsearch_index():
    body = {
        'mappings': {
            'properties': {
                'title_name': {'type': 'keyword'},
                'title_vector': {
                    'type': 'dense_vector',
                    'dims': 1000
                }
            }
        }
    }
    es.indices.create(index = 'document_recognition', body = body)

def get_feature_map_encode(img):
    ORIGIN_IMAGE_SIZE = (224,224)
    input = img_to_array(img) 
    input = smart_resize(input, ORIGIN_IMAGE_SIZE, interpolation='bilinear')
    input = np.expand_dims(input, axis=0)
    feature_maps = model.predict(input)
    feature_maps_list = feature_maps[0].tolist()
    return feature_maps_list


def create_elasticsearch_datasets(files, title_name):
    index = 1
    for image_path in files:
        img = load_img(image_path)
        feature_maps_list = get_feature_map_encode(img)
        doc = {'title_name': image_path, 'title_vector': feature_maps_list}
        es.create('document_recognition', id = index, body = doc)
        index += 1
    


ds_vietname_cmnd = glob.glob("./datasets/identity_card/vietnam_cmnd/*.png")
ds_vietname_cancuoc = glob.glob("./datasets/identity_card/vietnam_cancuoc/*.png")
ds_discharge_record = glob.glob("./datasets/hopital/discharge_record/*.png")

create_elasticsearch_datasets(ds_vietname_cmnd, 'viet nam cmnd')
create_elasticsearch_datasets(ds_vietname_cancuoc, 'viet nam can cuoc')
create_elasticsearch_datasets(ds_discharge_record, 'discharge record')

