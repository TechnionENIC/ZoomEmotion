import queue
# import threading
import threading

import cv2
import numpy as np
import pyautogui
from mss import mss
from EmotionDetection import IEmotionDetection
from FrameDetection import FrameDetection
from nameDetection import NameDetection

from PIL import Image
import face_recognition
import time
import os
import random
import sys


class FaceData:
    def __init__(self, emotion="empty", score=-1.0, left=-1, right=-1, up=-1, down=-1, name="", ind=-1):
        self.emotion = emotion.lower()
        self.score = score
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.name = name
        self.ind = ind


returnEmpty = FaceData()


def capture_screen1(screen_number):
    frame = None
    filename = ""
    gray = []
    with mss() as sct:
        filename = sct.shot(mon=screen_number, output="sct.png")
        img = cv2.imread(filename)
        frame = np.array(img)  # convert these pixels to a proper numpy array to work with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray


def capture_screen(sn):
    img_path = "screenshot.png"
    img = pyautogui.screenshot()
    # img = cv2.imread(img_path)
    img = np.array(img)  # convert these pixels to a proper numpy array to work with OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray



class FaceDetection(object):

    def __init__(self):
        cascPath = "haarcascade_frontalface_default.xml"
        cascAltPath = "haarcascade_frontalface_alt2.xml"
        # casc_dir = cv2.data.haarcascades
        casc_dir = "CascadeFiles"
        self.face_cascade = cv2.CascadeClassifier(casc_dir + "/" + cascPath)

    def detect(self, img):
        if img is None:
            print("Can't open image file")
            return returnEmpty
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img) == 0 or len(img[0]) == 0:
            return []

        img = cv2.cvtColor(img, cv2.CV_8U)
        faces = self.face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if faces is None or len(faces) == 0:
            # print('Failed to detect face')
            return []
        return faces


"""
this is the image processing part. 
it contain frame detection, text detection, face detection and emotion detection to do all the work.
first it take a screenshot, then use all the above to return faces with their names and emotions that were in this
screenshot.
"""


class NamesAndEmotionRecognition:
    def __init__(self, screen_number, emotion_detection: IEmotionDetection, num_threads=2):
        self.frames_to_detect = queue.Queue()
        self.face_detector = FaceDetection()
        self.name_detector = NameDetection('heb')
        self.screen_number = screen_number
        self.num_threads = num_threads
        self.frame_detector = FrameDetection()
        self.emotion_detector = emotion_detection
        self.detected_faces = queue.Queue()
        self.threading_pool = []

        self.text_height = 34
        self.image = []
        """for _ in range(self.num_threads):
            self.threading_pool.append(threading.Thread(target=self.face_thread_work, args=(), daemon=True))
        for thread in self.threading_pool:
            thread.start()"""

    def set_score_threshold(self, score_threshold):
        self.emotion_detector.set_score_threshold(score_threshold)

    def set_screen_number(self, screen_number):
        self.screen_number = screen_number

    def _detection_emotion_from_face(self, face, img):      
        x, y, w, h = face
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        ny = round(ny - 20)  # fix wrong offset        
        face_img = img[y:y + h, x:x + w]
        emotion = "Neutral"
        reshaped_img = face_img.reshape(face_img.shape[0], face_img.shape[1], 1) 
        encoding_1  = face_recognition.face_encodings(reshaped_img.repeat(3,2))

        np.set_printoptions(threshold=sys.maxsize)
        if len(encoding_1) != 0:
            emotion, score = self.emotion_detector.detect(face_img)
            print(score)
            #img11 = Image.fromarray(face_img, 'L')
            #curr_time = time.time()
            # img11.save(os.path.join(r'C:\Users\neuro\Desktop\Zoom_Emotion\all_photos',str(curr_time) + ".png"))
            # file = open(os.path.join(r'C:\Users\neuro\Desktop\Zoom_Emotion\all_photos',"emotion_ " + str(curr_time) + ".txt"), "w+")
            # file.write("Emotion: " + emotion + " score: " + str(score))
            # file.close()
            print(emotion)
        else:
            score = 0
        return emotion, score, nx, nx + nr, ny, ny + nr

    def face_thread_work(self, frame):
        # frame = self.frames_to_detect.get()
        image = self.image
        face_data = self.detect_face_from_frame(frame, image)
        self.detected_faces.put(face_data)

    def detect_face_from_frame(self, frame_rect, img):
        x, y, w, h = frame_rect
        face = None
        frame_data = img[y:y + h, x:x + w]
        #img11 = Image.fromarray(frame_data, 'L')
        #img11.save(os.path.join(r'C:\Users\neuro\Desktop\ZoomEmotion\all_photos',str(time.time()) + ".png"))
        faces = self.face_detector.detect(img)
        if len(faces) > 0:
            face = faces[random.randint(0,len(faces) - 1)]
        text_place = img[y + h - self.text_height:y + h, x:x + int(w / 2)]
        # text_place = frame_data
        if len(text_place) == 0:
            return FaceData(name="", emotion="faceless")
        name = self.name_detector.get_name_in_image(text_place)
        if face is not None:
            #face = self.fix_face_coords(face, frame_rect)
            emotion, score, left, right, up, down = self._detection_emotion_from_face(face, img)
            face_data = FaceData(emotion, score, left, right, up, down, name=name)
        else:
            face_data = FaceData(name="", emotion="faceless")
        return face_data

    def detect_faces_from_screen(self):
        self.image = capture_screen(self.screen_number)

        rects = self.frame_detector.detect_frames(self.image)
        if len(rects) == 0:
            self.detected_faces.queue.clear()
            return []
        for rect in rects:
            # for thread work
            # self.frames_to_detect.put(rect, block=True)
            # threading.Thread(target=self.face_thread_work, args=[rect], daemon=True, ).start()
            self.face_thread_work(rect)

        # wait for threads to finish
        # while len(self.detected_faces.queue) < len(rects):
        # continue
        detected_faces = list(self.detected_faces.queue)
        self.detected_faces.queue.clear()
        return detected_faces

    def destroy(self):
        pass
        # for _ in range(len(self.threading_pool)):
        #   self.frames_to_detect.put(None)

    def fix_face_coords(self, face, frame_rect):
        fx, fy, w, h = face
        rx, ry, rw, rh = frame_rect
        fixed_face = rx + fx, ry + fy, w, h
        return fixed_face
