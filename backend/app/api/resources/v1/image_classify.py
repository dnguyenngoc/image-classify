from fastapi import Depends, APIRouter, Form, UploadFile, HTTPException, File
from sqlalchemy.orm import Session
from databases.db import get_db
from fastapi.responses import StreamingResponse

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
import os
import uuid
import glob



router = APIRouter()


@router.get('/images/{file_name}')
async def get_images(file_name: str):
    def iterfile():  
        with open('./images/'+file_name, mode="rb") as file_like:  
            yield from file_like  
    return StreamingResponse(iterfile(), media_type="image/png")


# @router.post("/predict")
# def predict( 
#     file: UploadFile = File(...),
# ):
#     byte_file = file.file.read()
#     if len(byte_file) > 20**22: 
#         raise HTTPException(status_code=400, detail="wrong size > 20MB")
#     image = cv2.imdecode(np.frombuffer(byte_file, np.uint8), 1)

#     image = scanner.process(image)
#     image = image_utils.pre_process(image)
#     text = tesseract.excecute(image)
#     tokens = word_tokenize(text_utils.text_cleaner(text))
#     tokens = text_utils.fix_tokens(tokens, config_ml.STOPWORDS)
#     if len(tokens) >= config_ml.LEN_TOKEN:
#         tokens = tokens[0:config_ml.LEN_TOKEN]
#     else:
#         for i in range(len(tokens) - config_ml.LEN_TOKEN):
#             tokens.append('None')
#     input_ids = torch.tensor([tokenizer.encode(tokens)])
#     with torch.no_grad():
#         features = phobert(input_ids) 
#     es_dim = features['pooler_output'][0].tolist()
#     pred = es_utils.matching_elasticsearch_index(es, config_ml.INDEX_NAME, es_dim)
#     hits = pred['hits']['hits']
#     score = hits[0]['_score']/2
#     class_pre = hits[0]['_source']
#     class_pre['score'] = score
#     return class_pre


@router.post("/predict")
def pre_processing(
    file: UploadFile = File(...),
):
    files = glob.glob('./images/*')
    for f in files:
        os.remove(f)
    uuidOne = uuid.uuid1()
    file_name = str(uuidOne) + file.filename
    byte_file = file.file.read()

    if len(byte_file) > 20**22: 
        raise HTTPException(status_code=400, detail="wrong size > 20MB")
    image = cv2.imdecode(np.frombuffer(byte_file, np.uint8), 1)
    image = scanner.process(image)
    text = tesseract.excecute(image)
    tokens = word_tokenize(text_utils.text_cleaner(text))
    image = image_utils.pre_process(image)
    cv2.imwrite('./images/' + file_name,  image)
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
    class_pre['pre_url'] = 'http://10.1.133.3:8082/api/v1/image-classify/images/'+ file_name
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
