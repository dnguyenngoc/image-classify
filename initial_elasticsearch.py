from elasticsearch import Elasticsearch
import glob
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
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
                'document_type_name': {'type': 'text'},
                'document_type_id': {'type': 'integer'},
                'image_url': {'type': 'text'},
                'vector': {
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


def create_elasticsearch_datasets(files, document_type_name, document_type_id, index):
    for image_path in files:
        img = load_img(image_path)
        feature_maps_list = get_feature_map_encode(img)
        document_type_id = 1
        doc = {
            'document_type_name': document_type_name, 
            'document_type_id': document_type_id, 
            'image_url': image_path,
            'vector': feature_maps_list,
        }
        es.create('document_recognition', body = doc, id = index)
        index += 1
    return index
        

# Create index    
create_elasticsearch_index()


# upload datasets to es
ds_vietname_cmnd = glob.glob("./datasets/identity_card/vietnam_cmnd/*.png")
ds_vietname_cancuoc = glob.glob("./datasets/identity_card/vietnam_cancuoc/*.png")
ds_discharge_record = glob.glob("./datasets/hopital/discharge_record/*.png")

ds_tp_bank = glob.glob("./datasets/bank/tp_bank/*.tif")
ds_invoice = glob.glob("./datasets/bank/invoice/*.jpg")
ds_payment_order = glob.glob("./datasets/bank/payment_order/*.tif")


index = 1
index = create_elasticsearch_datasets(ds_vietname_cmnd, 'viet nam cmnd', 1, index)
print("[RUN] ds_vietname_cmnd with: {}".format(len(ds_vietname_cmnd)))

index = create_elasticsearch_datasets(ds_vietname_cancuoc, 'viet nam can cuoc', 2, index)
print("[RUN] ds_vietname_cancuoc with: {}".format(len(ds_vietname_cancuoc)))

index = create_elasticsearch_datasets(ds_discharge_record, 'discharge record', 3, index)
print("[RUN] ds_discharge_record  with: {}".format(len(ds_discharge_record)))

index = create_elasticsearch_datasets(ds_tp_bank, 'tp bank form', 4, index)
print("[RUN] ds_tp_bank  with: {}".format(len(ds_tp_bank)))

index = create_elasticsearch_datasets(ds_invoice, 'invoice', 5, index)
print("[RUN] ds_invoice  with: {}".format(len(ds_invoice)))

index = create_elasticsearch_datasets(ds_payment_order, 'payment_order', 6, index)
print("[RUN] ds_payment_order  with: {}".format(len(ds_payment_order)))

print("[DONE]")
