
import cv2
import os
from PIL import Image
from pprint import pprint


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


from helpers.tesseract_utils import Tesseract
from helpers import text_utils


tesseract = Tesseract(out_type='string')


image_path = '/home/pot/project/image-classify/datasets/text_segmentation/out/document_classify/discharge_record/1.png'

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = tesseract.excecute(gray)



tokens = word_tokenize(text_utils.text_cleaner(text))


stopwords = set([
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

tokens = [ item for item in tokens  if item not in stopwords and item.isnumeric() == False]

print(tokens)
# text = text[0:100]


