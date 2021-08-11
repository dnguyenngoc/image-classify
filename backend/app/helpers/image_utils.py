import glob
import cv2
from torchvision.transforms import transforms
import numpy as np
# from scipy.ndimage import interpolation as inter
from helpers.skew import SkewDetect
import PIL
from PIL import ImageDraw
from PIL import Image
from matplotlib import cm
import numpy as np

skew = SkewDetect()


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


# def get_title(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     (thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # convert2binary
#     mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
#     (contours, _) = cv2.findContours(~binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
#     heights = [cv2.boundingRect(contour)[3] for contour in contours] # collecting heights of each contour
#     avgheight = sum(heights)/len(heights) # average height
#     for c in contours:
#         [x,y,w,h] = cv2.boundingRect(c)
#         if h > 2*avgheight:
#             cv2.drawContours(mask, [c], -1, 0, -1)
#     x, y = mask.shape
#     value = max(math.ceil(x/100),math.ceil(y/100))+10 #heuristic
#     mask = rlsa.rlsa(mask, True, False, value) #rlsa application
#     (contours, _) = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours
#     mask2 = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
#     for contour in contours:
#         [x, y, w, h] = cv2.boundingRect(contour)
#         if w > 0.60*image.shape[1]: # width heuristic applied
#             title = image[y: y+h, x: x+w] 
#             mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
#             image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
#     return mask2, image


def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    max_index = 0
    max_check = 0
    for i in range(len(bounds)):
        bound = bounds[i]
        p0, p1, p2, p3 = bound[0]
        if p3[1] - p0[1] > max_check:
            max_check = p3[1] - p0[1]
            max_index = i
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image, max_index


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
    image = cv2.imread(path)
    return image


def load_pil(path):
    im = PIL.Image.open(path)
    return im


def to_pil_image(image):
    im_box = Image.fromarray(image)
    return im_box


def pre_process(img):
    img = skew.determine_skew(img)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Converting to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #Removing Shadows
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)
    
    #Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)#increases the white region in the image 
    img = cv2.erode(img, kernel, iterations=1) #erodes away the boundaries of foreground object
    
#     Apply blur to smooth out the edges
#     img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply threshold to get image with only b&w (binarization)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    imgf = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    return imgf
    

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 