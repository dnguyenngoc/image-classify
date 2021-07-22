import cv2
import os
import numpy as np
from numpy.core.fromnumeric import resize

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

class Scanner:
    def __init__(self, input, output):
        self.image = cv2.imread(input)
        
    def sort_contours(self, cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)


    def gray(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


    def process_edge(self, image, open_k_size =11, close_k_size = 11, is_gaussian_blur = True):
        if open_k_size > 0:
            kernel = np.ones((open_k_size, open_k_size),np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        if close_k_size > 0:
            kernel = np.ones((close_k_size, close_k_size),np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        if is_gaussian_blur:
            image = cv2.GaussianBlur(image,(7,7),0)
        edges = cv2.Canny(image,50,60)
        return edges


    def find_contours(self, image, is_retr_list = True):
        if is_retr_list:
            contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy


    def distance(self, p1, p2):
        return ((((p1[0] - p2[0] )**2) + ((p1[1]-p2[1])**2) )**0.5)

    def four_point_transform(self, image, approx): 
        wa = self.distance(approx[2][0], approx[3][0])
        wb = self.distance(approx[1][0], approx[0][0])
        mw = max(wa, wb)

        ha = self.distance(approx[1][0], approx[2][0])
        hb = self.distance(approx[0][0], approx[3][0])
        mh = max(ha, hb)

        src_ = np.float32([[approx[i][0][0], approx[i][0][1]] for i in range(4)])
        dst_ = np.float32([[0,0], [mw-1, 0], [mw-1, mh-1], [0, mh-1]])

        m = cv2.getPerspectiveTransform(src_, dst_)
        out = cv2.warpPerspective(image, m, (int(mw), int(mh)), cv2.BORDER_REPLICATE, cv2.INTER_LINEAR)
        return out

    def process(self):
        image = self.image.copy()
        h, w = image.shape[:2]
        ratio = h/500

        image = cv2.resize(image, (int(w*ratio), int(h*ratio)), interpolation = cv2.INTER_LINEAR)
        image_gray = self.gray(image)
        edge = self.process_edge(image_gray, 11, 11, True)
        edge_cache = edge.copy()

        thresh = 100
        ret,thresh_img = cv2.threshold(edge, thresh, 255, cv2.THRESH_BINARY)
        contours, hierarchy = self.find_contours(thresh_img)
        sum_area = 0
        mean_area = 0
        hull = []
        shapes = []


        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))
            sum_area += cv2.contourArea(hull[i])

        mean_area = sum_area / len(hull)

        for i in range(len(hull)):
            if cv2.contourArea(hull[i]) >= mean_area:
                shapes.extend(hull)

        shapes, bounding_boxes = self.sort_contours(shapes)
        new_shapes = []
        for i in range(len(shapes)):
            for j in range(len(shapes[i])):
                new_shapes.extend(shapes[i][j])

        hull[0] = cv2.convexHull(shapes[0], hull[0], False)

        drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)
        cv2.drawContours(drawing, hull, 0, 255, 2)

        thresh_img = self.gray(drawing)
        contours, hierarchy = self.find_contours(thresh_img)

        approx = []
        for i in range(len(contours)):
            peri = 0.01*cv2.arcLength(contours[i], True)
            approx.append(cv2.approxPolyDP(contours[i], peri, True))
        
        for i in range(len(approx)):
            if len(approx[i]) == 4:
                for j in range(len(approx[i])):
                    approx[i][j] =  approx[i][j]*ratio
                output = self.four_point_transform(image, approx[i])
        show(output)



scanner = Scanner('.\\data\\007.png', '..\out')
image = scanner.process()