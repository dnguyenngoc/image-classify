# ++++++++++++++++++++++++++++++++++++++++++++ LOAD DEEP LEARN ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from transformers import AutoModel, AutoTokenizer
from helpers.tesseract_utils import Tesseract
from elasticsearch import Elasticsearch
from settings import config_ml
from helpers import scanner


phobert = AutoModel.from_pretrained("models/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("models/phobert-base")
tesseract = Tesseract(out_type='string')
es = Elasticsearch([{'host': config_ml.ES_HOST, 'port': config_ml.ES_PORT}])
scanner = scanner.ScannerFindContours()
