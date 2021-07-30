

LEN_FEATURE = 768
LEN_TOKEN = 140
INDEX_NAME = 'image-classify'
ES_HOST = '10.1.32.130'
ES_PORT = '9200'

CLASSES = {
    1: 'discharge record',
    2: 'driver licence',
    3: 'invoice',
    4: 'resume',
    5: 'vehicle certificate',
    6: 'degree of bachelor' 
}

STOPWORDS = set(['\\', '(', ')', ':', '.', ';', ',', '\\\\', '\\\\\\', '-', '%', '`', '—-', '?', '——', '--', '@',  '[', ']', '.....', '``', 'đụ', 'đéo'])