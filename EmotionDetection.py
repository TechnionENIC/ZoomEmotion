from enum import Enum

from RonNet import RonNetWrap


class IEmotionDetection:
    # those emotion will be draw on screen.
    # you need to make sure there are amoji in the emoji folder for each emotion
    emotion_to_draw = None
    # used for log action
    all_emotion = None

    def detect(self, img):
        raise Exception()


class RonNetEmotionDetection(IEmotionDetection):
    def __init__(self):
        self.detector = RonNetWrap()
        #RONNET_NEG_EMOTION = ['Angry', 'Disgusted', 'Fearful', 'Sad', 'Surprised']
        RONNET_NEG_EMOTION = ['Angry','Fearful', 'Sad']
        RONNET_EMOTION = RONNET_NEG_EMOTION + ['Happy', 'Neutral']
        EMOTION_TO_DRAW = RONNET_EMOTION

        self.emotion_to_draw = EMOTION_TO_DRAW
        self.all_emotion = RONNET_EMOTION

    def detect(self, face_img):
        emotion = ""
        score = -1
        try:
            emotion, score = self.detector.top_emotion(face_img)
        except:
            # print("no face")
            # cv2.imshow("img",img)
            pass
        return emotion, score