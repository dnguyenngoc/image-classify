from helpers.image_utils import show, write, load, load_datasets
from helpers.scanner import ScannerFindContours

import cv2


if __name__ == "__main__":
    scanner = ScannerFindContours()

    invoice = load_datasets('./datasets/document_classify/train/bank/invoice')
    invoice_out = './datasets/text_segmentation/out/document_classify/invoice'
    for item in invoice:
        out_name = item.split("\\")[-1]
        image = scanner.process(load(item))
        write(invoice_out + '\\' + out_name.replace(".tif", '.png').replace(".pngf", '.png').replace('.jpg', '.png'), image)
        # write(invoice_out + '\\' + out_name, image)

    # dir = "C:/Users/duynn_1/Desktop/cmnd"
    # invoice = load_datasets(dir)
    # for item in invoice:
    #     name = item.split("\\")[-1]
    #     image = load(item)
    #     write('./datasets/document_classify/train/bank/invoice/' + name.replace('.jpg', '-in.png'), image)
