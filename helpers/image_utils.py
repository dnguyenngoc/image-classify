import glob
import cv2
from torchvision.transforms import transforms

def load_datasets(path: str, type: str ='\\*'):
    return glob.glob(path + type)


def load_image_transform(image_path):
    image = cv2.imread(image_path)
    rat = 1000 / image.shape[0]
    image = cv2.resize(image, None, fx=rat, fy=rat)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = transform(image)
    return image