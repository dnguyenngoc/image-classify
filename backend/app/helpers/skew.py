import os
from helpers import image_utils
import matplotlib.pyplot as plt
plt.figure()

""" Calculates skew angle """
import os
import imghdr
import optparse

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
import cv2
from PIL import Image as im
from scipy.ndimage import interpolation as inter


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        sigma=3.0,
        num_peaks=20,
    ):

        self.sigma = sigma
        self.num_peaks = num_peaks

    def get_max_freq_elem(self, arr):

        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr

    def compare_sum(self, value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False


    def calculate_deviation(self, angle):
        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)

        return deviation


    def determine_skew(self, image):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = canny(gray, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)

        if len(ap) == 0:
            return {"Image File": img_file, "Message": "Bad Quality"}

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = int(90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = int(ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = int(-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = int(90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            l = len(angles[j])
            if l > lmax:
                lmax = l
                maxi = j
        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])
            ans_res = np.mean(ans_arr)

        else:
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)
        if (70 < ans_res <= 90):
            ans_res = -(90-ans_res)
        elif (-90 <= ans_res < -70):
            ans_res =  90 + ans_res
        else:
            ans_res = 0

        data = inter.rotate(image, ans_res, reshape=False, order=0)
#         img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
#         (h, w) = gray.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, ans_arr, 1.0)
#         rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
#           borderMode=cv2.BORDER_REPLICATE)
        return data
    