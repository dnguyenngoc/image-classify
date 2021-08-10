import pytesseract 
from pytesseract import Output


class Tesseract:
    def __init__(self, out_type: str = 'string', psm: str = '--psm 6'):
        self.out_type = out_type

    def excecute(self, gray):
        if self.out_type == 'string':
            return pytesseract.image_to_string(gray, lang='vie', config='--psm 6')
        elif self.out_type == 'dict':
            return pytesseract.image_to_data(gray, lang='vie', output_type=Output.DICT, config='--psm 6')
