

LEN_FEATURE = 768
LEN_TOKEN = 150
INDEX_NAME = 'image-classify'
ES_HOST = '10.1.32.130'
ES_PORT = '9200'

CLASSES = {
    1: 'discharge record',
    2: 'driver licence',
    # 3: 'invoice',
    4: 'resume',
    5: 'vehicle registration certificate',
    6: 'degree of bachelor',
    7: 'test'
}

STOPWORDS = set([
    '\\', '(', ')', ':', '.', ';', ',', '\\\\', '\\\\\\', '-', '%', '`', '—-', '?', '——', '--', '@',  '[', ']', '.....', '``', 'đụ', 'đéo'
    'cộng', 'hòa', 'xã', 'hội', 'chủ', 'nghĩa', 'việt', 'nam', 'độc', 'lập', 'tự', 'do', 'hạnh', 'phúc',
    'hĩa', 'phỏ', 'chũ'
])