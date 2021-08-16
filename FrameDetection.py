import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_rect_area_from_shape(rect_shape):
    xl = rect_shape[0][0]
    yh = rect_shape[0][1]
    xh = rect_shape[3][0]
    yl = rect_shape[3][1]
    return xl, yl, abs(xh - xl), abs(yh - yl)


def calc_rect_area(rect):
    x, y, w, h = rect
    return float(w * h)


class FrameDetection:

    def __init__(self):
        self.min_area_size = 20_000
        self.rects = []
        self.min_area_diff_ratio = 0.6
        self.max_rect_size_compare_to_image = 0.6

    def detect_frames(self, image):
        return self._detect_frames(self.negegate_image(image))

    def _detect_frames(self, image):
        edge = cv2.Canny(image, 140, 210)
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rect_list = [cv2.boundingRect(cnt) for cnt in contours]

        image_size = self.calc_image_size(image)
        relevant_rects = [rect for rect in rect_list if
                          image_size * self.max_rect_size_compare_to_image >= calc_rect_area(
                              rect) >= self.min_area_size]
        if len(relevant_rects) == 0:
            return []
        sorted_rects = list(sorted(relevant_rects, key=lambda rect: calc_rect_area(rect), reverse=True))
        max_rect_area = calc_rect_area(sorted_rects[0])
        biggest_rects = [rect for rect in sorted_rects if
                         calc_rect_area(rect) / max_rect_area >= self.min_area_diff_ratio]
        return biggest_rects

    def draw_rects(self, image, rects):

        for x, y, w, h in rects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        plt.imshow(image)

    def calc_image_size(self, image):
        return len(image) * len(image[0])

    def negegate_image(self, image):
        black_range = range(0, 30)
        black = 0
        white = 255
        yellow_range = range(219, 239)
        return np.array(pd.DataFrame(image).apply(
            lambda row: row.apply(lambda pixel: black if pixel in black_range or
                                                        pixel in yellow_range else white))).astype(np.uint8)


if __name__ == "__main__":
    fd = FrameDetection()
    img_dir = "images"
    file_name = "WhatsApp Image 2021-02-25 at 16.57.34.jpeg"
    file2 = "childes_zoom.jpeg"
    children_file = "childreans.PNG"
    children_file2 = "childes2.PNG"
    many = "many_peoples.png"
    full = "Capture.PNG"
    frame = "../final presentation/frame.PNG"
    yellow_frame = "../final presentation/yellow_fram.PNG"
    img = img_dir + "/" + full
    image = cv2.imread(img)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = fd.detect_frames(gray)
    #neg_img = fd.negegate_image(gray)
    fd.draw_rects(image, rects)
