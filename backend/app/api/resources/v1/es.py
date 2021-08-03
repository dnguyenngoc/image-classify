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
import uuid

router = APIRouter()



@router.post("/indices")
def create(
    *,
    file: UploadFile = File(...),
    name: str = Form(...),
    class_id: int = Form(...),
    stt: int = Form(...),
    url: str = Form(...)
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
    
    try:
        _id = name + str(class_id) + str(stt)
        _id = uuid.uuid5(uuid.NAMESPACE_URL, _id)
        create = es_utils.create_elasticsearch_datasets(es, config_ml.INDEX_NAME,class_id,name, url, es_dim, _id)
        return 'ok!'
    except Exception as e:
        return str(e)


@router.delete("/indices/{class_id}")
def delete(
    class_id: str
):
    return es_utils.delete_elasticsearch_doc_by_class_id(es, config_ml.INDEX_NAME, class_id)


@router.get("/indices/{class_id}")
def get(
    class_id: str
):
    data = es_utils.get_elasticsearch_doc_by_class_id(es, config_ml.INDEX_NAME, class_id)
    return data['hits']['hits']