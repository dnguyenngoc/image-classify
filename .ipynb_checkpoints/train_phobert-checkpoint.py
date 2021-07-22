import torch
from transformers import AutoModel, AutoTokenizer
from pprint import pprint
from helpers.tesseract_utils import Tesseract
import pandas as pd
import cv2
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from helpers import text_utils
from helpers import es_utils
from elasticsearch import Elasticsearch
from sklearn.utils import shuffle


LEN_FEATURE = 768
INDEX_NAME = 'image-classify'
ES_HOST = '10.1.32.130'
ES_PORT = '9200'
PATH_DATASET = './datasets/image_classify/main.xlsx'
PATH_TEST = './datasets/image_classify/test.xlsx'


STOPWORDS = set([
    "ai", "bằng", "bị", "bộ", "cho", "chưa", "chỉ", "cuối", "cuộc",
    "các", "cách", "cái", "có", "cùng", "cũng", "cạnh", "cả", "cục",
    "của", "dùng", "dưới", "dừng", "giữa", "gì", "hay", "hoặc",
    "khi", "khác", "không", "luôn", "là", "làm", "lại", "mà", "mọi",
    "mỗi", "một", "nhiều", "như", "nhưng", "nào", "này", "nữa",
    "phải", "qua", "quanh", "quá", "ra", "rất", "sau", "sẽ", "sự",
    "theo", "thành", "thêm", "thì", "thứ", "trong", "trên", "trước",
    "trừ", "tuy", "tìm", "từng", "và", "vài", "vào", "vì", "vẫn",
    "về", "với", "xuống", "đang", "đã", "được", "đấy", "đầu", "đủ",
    '', ' ', '^', '_', '-', '|', '(', ')', ':', ';', ',', '~', '<', '.','⁄', '/⁄', '{', '/', '⁄', '}', '—', '@'
])

CLASSES = {
    1: 'discharge record',
    2: 'identity card front',
    3: 'identity card back',
    4: 'payment order form',
}

# load tool
phobert = AutoModel.from_pretrained("models/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("models/phobert-base")
tesseract = Tesseract(out_type='string')
es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT}])

es_utils.delete_elasticsearch_index(es, INDEX_NAME)
es_utils.create_elasticsearch_index(es, INDEX_NAME, LEN_FEATURE)


#####################################################################################################################
df = pd.read_excel(PATH_DATASET)

index = 1
for i in range(len(df['image'])):
    image_path = df['image'][i]
    print('[run]: ', image_path)
    class_id = df['class'][i]
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = tesseract.excecute(gray)

    tokens = word_tokenize(text_utils.text_cleaner(text))
    tokens = [item for item in tokens  if item not in STOPWORDS and item.isnumeric() == False]

    if len(tokens) >= 50:
        tokens = tokens[0:50]
    else:
        for i in range(len(tokens) - 50):
            tokens.append('None')

    input_ids = torch.tensor([tokenizer.encode(tokens)])
    with torch.no_grad():
        features = phobert(input_ids) 
    

    es_dim = features['pooler_output'][0].tolist()

    es_utils.create_elasticsearch_datasets(es, INDEX_NAME, class_id, CLASSES[class_id] ,image_path, es_dim, index)

    index += 1
#####################################################################################################################


#####################################################################################################################
df = pd.read_excel(PATH_TEST)
x = 0
for i in range(len(df['image'])):
    image_path = df['image'][i]
    print(image_path)
    class_id = df['class'][i]
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = tesseract.excecute(gray)
    tokens = word_tokenize(text_utils.text_cleaner(text))
    tokens = [item for item in tokens  if item not in STOPWORDS and item.isnumeric() == False]
    if len(tokens) >= 50:
        tokens = tokens[0:50]
    else:
        for i in range(len(tokens) - 50):
            tokens.append('None')
    input_ids = torch.tensor([tokenizer.encode(tokens)])
    with torch.no_grad():
        features = phobert(input_ids) 
    es_dim = features['pooler_output'][0].tolist()
    pred = es_utils.matching_elasticsearch_index(es, INDEX_NAME, es_dim)
    if class_id == pred['hits']['hits'][0]['_source']['id']:
        x += 1
    else:
        print(class_id, pred['hits']['hits'][0]['_source']['id'])
    # print(image_path, class_id, pred['hits']['hits'][0]['_source']['id'])
print(x/len(df['image']))
#####################################################################################################################



    

