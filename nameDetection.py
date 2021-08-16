import time

import cv2
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'


class NameDetection:
    def __init__(self, lang='heb+eng'):
        self.lang = lang

    def get_name_in_image(self, image):
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        imgData = pytesseract.image_to_data(thresh, lang=self.lang, output_type='data.frame',
                                            pandas_config='{encoding="ISO_8859_8"}')
        imgDataProc = imgData[['text', 'word_num']]
        imgDataProc['text'] = imgDataProc['text'].map(str)
        imgDataProc = imgDataProc.dropna()
        imgDataProc = imgDataProc[imgDataProc.text.str.contains(r'[0-9A-Za-zא-ת]')]
        name = ""
        for word in imgDataProc['text']:
            if word != "nan":
                name += str(word) + " "
        return name

    def attach_names_to_faces(self, image, faces):
        if len(faces) == 0:
            return []
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        start=time.time()
        imgData = pytesseract.image_to_data(thresh, lang='heb+eng', output_type='data.frame',
                                            pandas_config='{encoding="ISO_8859_8"}')
        #print(time.time() -start)
        imgDataProc = imgData[['top', 'left', 'height', 'text']]
        imgDataProc = imgDataProc.rename(columns={"top": "Top", "left": "Left", "height": "Height", "text": "Text"})
        imgDataProc["Bottom"] = imgDataProc["Top"] + imgDataProc["Height"]
        imgDataProc = imgDataProc.dropna()
        imgDataProc = imgDataProc[imgDataProc.Text.str.contains(r'[A-Za-zא-ת]')]
        imgDataProc = imgDataProc[['Bottom', 'Left', 'Text']]
        i = 1
        for (x, y, w, h) in faces:
            """imgDataProc["distance{}".format(i)] = np.where(
            ((imgDataProc["Left"] < x) & (imgDataProc["Bottom"] > y)),
            ((imgDataProc["Left"] - x) ** 2 + (imgDataProc["Bottom"] - y) ** 2), 'NaN')"""
            imgDataProc["distance{}".format(i)] = ((imgDataProc["Left"] - x) ** 2 + (imgDataProc["Bottom"] - y) ** 2)
            i += 1
        minOfEachCol = imgDataProc.min(skipna=True)
        minDists = []
        minDistsCol = []

        for i in minOfEachCol.index:
            if i != 'Bottom' and i != 'Left' and i != 'Text':
                minDists.append(minOfEachCol[i])
            minDistsCol.append(i)
        relevantData = pd.DataFrame()
        for i in range(len(minDists)):
            row = imgDataProc[imgDataProc[minDistsCol[i + 3]].isin([minDists[i]])]
            if relevantData.empty:
                relevantData = pd.DataFrame(row, columns=minDistsCol)
            else:
                relevantData = relevantData.append(row)

        finalData = relevantData['Text']
        return finalData
