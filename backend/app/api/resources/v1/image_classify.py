from fastapi import APIRouter, UploadFile, HTTPException, File
from fastapi.responses import StreamingResponse

from ml  import  phobert, es, tokenizer, reader
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from settings import config_ml
from helpers import text_utils
from helpers import es_utils
from helpers import image_utils
import torch
from settings import config

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


@router.post("/predict")
async def pre_processing(
    file: UploadFile = File(...),
):
    files = glob.glob('./images/*')
    for f in files:
        os.remove(f)
    uuidOne = uuid.uuid1()
    file_name = str(uuidOne) + file.filename
    byte_file = file.file.read()

    if len(byte_file) > 5**22: 
        raise HTTPException(status_code=400, detail="wrong size > 5MB")
    image = cv2.imdecode(np.frombuffer(byte_file, np.uint8), 1)

    bounds = reader.readtext(image)

    pil_image = image_utils.to_pil_image(image)
    pil_image, max_index = image_utils.draw_boxes(image = pil_image, bounds=bounds)
    # cv2.imwrite('./images/' + file_name,  pil_image)
    pil_image.save('./images/' + file_name)

    texts = bounds[max_index][1]
    for item in bounds:
        box, text, score = item
        texts += ' '+ text

    texts = text_utils.text_cleaner(texts)
    tokens = word_tokenize(texts)
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
    if score < 0.92:
        a = {'0': 0, '1':0, '2': 0}
        for i in range(len(hits)):
            a[str(i)] += 1
        max_end = max(a, key=a.get)
        class_pre = hits[int(max_end)]['_source']
        score = hits[int(max_end)]['_score']/2
    return {
        'id': class_pre['id'],
        'name': class_pre['name'],
        'pre_url': 'http://{}:8082/api/v1/image-classify/images/'.format(config.HOST_NAME) + file_name,
        'score': score,
        'texts': texts.replace("   ", " ").replace("  ", " "),
        'token': tokens,
        'es_dim': es_dim,
        'vector': class_pre['vector']
    }

    
# @router.post("/ocr/machine-writing")
# def ocr(
#     file: UploadFile = File(...),
# ):
#     byte_file = file.file.read()
#     if len(byte_file) > 20**22: 
#         raise HTTPException(status_code=400, detail="wrong size > 20MB")
#     image = cv2.imdecode(np.frombuffer(byte_file, np.uint8), 1)

#     # image = image_utils.pre_process(image)
#     bounds = reader.readtext(image)

#     texts = ''
#     for item in bounds:
#         box, text, score = item
#         texts += ' '+ text

#     return text_utils.text_cleaner(text)
