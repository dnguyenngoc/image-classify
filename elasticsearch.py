from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
import numpy as np
from elasticsearch import Elasticsearch
from pprint import pprint
import glob


# Load VGG16
model = VGG16()
model.summary()


# Load Elasticsearch
es = Elasticsearch([{'host': '10.1.32.130', 'port': '9200'}])


def get_feature_map_encode(img):
    ORIGIN_IMAGE_SIZE = (224,224)
    input = img_to_array(img) 
    input = smart_resize(input, ORIGIN_IMAGE_SIZE, interpolation='bilinear')
    input = np.expand_dims(input, axis=0)
    feature_maps = model.predict(input)
    feature_maps_list = feature_maps[0].tolist()
    # tensor = tf.convert_to_tensor(feature_maps)
    return feature_maps_list


files  = glob.glob("./test_data/*")
for file in files:
    vector = get_feature_map_encode(load_img(file))
    query = {
        'size': 1,
        'query': {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'vector') + 1",
                    "params":{
                        "queryVector": vector
                    }
                }
            }
        }
    }
    res = es.search(index='document_recognition', body = query)
    print(file.split("//")[-1], res['hits']['hits'][0]['_source']['document_type_name'])

