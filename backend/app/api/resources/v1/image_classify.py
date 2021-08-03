from fastapi import Depends, APIRouter, Form, UploadFile, HTTPException, File
from sqlalchemy.orm import Session
from databases.db import get_db


from ml  import scanner, phobert, es, tesseract, tokenizer
from helpers import image_utils
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from settings import config_ml
from helpers import text_utils
from helpers import es_utils
import torch

import cv2
import numpy as np


router = APIRouter()


@router.post("/predict")
def predict( 
    file: UploadFile = File(...),
):
    byte_file = file.file.read()
    if len(byte_file) > 20**22: 
        raise HTTPException(status_code=400, detail="wrong size > 20MB")
    image = cv2.imdecode(np.frombuffer(byte_file, np.uint8), 1)

    image = scanner.process(image)
    image = image_utils.pre_process(image)
    text = tesseract.excecute(image)
    tokens = word_tokenize(text_utils.text_cleaner(text))
    tokens = text_utils.fix_tokens(tokens, config_ml.STOPWORDS)
    if len(tokens) >= config_ml.LEN_TOKEN:
        tokens = tokens[0:config_ml.LEN_TOKEN]
    else:
        for i in range(len(tokens) - config_ml.LEN_TOKEN):
            tokens.append('None')
    input_ids = torch.tensor([tokenizer.encode(tokens)])
    with torch.no_grad():
        features = phobert(input_ids) 
    es_dim = features['pooler_output'][0].tolist()
    pred = es_utils.matching_elasticsearch_index(es, config_ml.INDEX_NAME, es_dim)
    hits = pred['hits']['hits']
    score = hits[0]['_score']/2
    class_pre = hits[0]['_source']
    class_pre['score'] = score
    return class_pre

    
@router.post("/ocr/machine-writing")
def ocr(
    file: UploadFile = File(...),
):
    byte_file = file.file.read()
    if len(byte_file) > 20**22: 
        raise HTTPException(status_code=400, detail="wrong size > 20MB")
    image = cv2.imdecode(np.frombuffer(byte_file, np.uint8), 1)
    image = image_utils.pre_process(image)
    text = tesseract.excecute(image)
    return text_utils.text_cleaner(text)
