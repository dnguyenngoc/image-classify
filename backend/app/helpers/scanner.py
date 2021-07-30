from numpy.core.fromnumeric import argmax
from helpers.image_utils import show
import cv2
import numpy as np
from scipy.spatial import distance as dist


class ScannerFindContours:
    def __init__(self):
        pass
        

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

    def order_points(self, pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]
        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")


    def four_point_transform(self, image, approx): 
        rect = self.order_points(approx)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
                            [0, 0],
                            [maxWidth - 1, 0],
                            [maxWidth - 1, maxHeight - 1],
                            [0, maxHeight - 1]
                        ], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), cv2.BORDER_REPLICATE, cv2.INTER_LINEAR)
        return warped


    def process(self, image):

        # pre image
#         image = image.copy()
        image_gray_origin = self.gray(image)
        h, w = image.shape[:2]
        ratio = h/500
        image = cv2.resize(image, (int(w/ratio), 500), interpolation = cv2.INTER_LINEAR)
        image_gray = self.gray(image)

        # hull function
        edge = self.process_edge(image_gray, 11, 11, True)
        thresh = 100
        _,thresh_img = cv2.threshold(edge, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = self.find_contours(thresh_img)
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
        shapes, _ = self.sort_contours(shapes)
        new_shapes = []
        for i in range(len(shapes)):
            for j in range(len(shapes[i])):
                new_shapes.extend(shapes[i][j])
        hull[0] = cv2.convexHull(shapes[0], hull[0], False)
        drawing = np.zeros((thresh_img.shape[0], thresh_img.shape[1], 3), np.uint8)
        cv2.drawContours(drawing, hull, 0, 255, 2)

        # get 4 point and crop
        new_thress = self.gray(drawing)
        contours, _ = self.find_contours(new_thress)
        approx = []
        for i in range(len(contours)):
            peri = 0.01*cv2.arcLength(contours[i], True)
            approx.append(cv2.approxPolyDP(contours[i], peri, True))
        check = False
        for i in range(len(approx)):
            if len(approx[i]) == 4:
                check = True
                approx_new = approx[i].reshape(-1,2)
                for k in range(len(approx_new)):
                    approx_new[k] = approx_new[k]*ratio
                output = self.four_point_transform(image, approx_new)
            
        if check == False or  output.shape[0] < 100:
            output = image
        return output
