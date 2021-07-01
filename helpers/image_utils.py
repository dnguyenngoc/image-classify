import glob
import cv2
from torchvision.transforms import transforms

def load_datasets(path: str, type: str ='/*'):
    return glob.glob(path  + type)

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


def show(img, name="disp", width=1000):
    """
    name: name of window, should be name of img
    img: source of img, should in type ndarray
    """
    cv2.namedWindow(name, cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(name, width, 1000)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write(filename, img):
    cv2.imwrite(filename, img)


def load(path):
    return cv2.imread(path)