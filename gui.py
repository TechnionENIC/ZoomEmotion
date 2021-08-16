import tkinter as tk
from win32api import GetSystemMetrics
import cv2
import logging as log
from FaceProcessing import NamesAndEmotionRecognition
from PIL import Image, ImageTk
from tkinter import ttk
from os import path
import time
from EmotionDetection import IEmotionDetection, RonNetEmotionDetection
from Timer import Timer
from ZELog import ZELog
from concurrent.futures import ThreadPoolExecutor

import warnings

warnings.filterwarnings("ignore")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)
EMPTY = "empty"



PINK = 86, 58, 203

APP_RUNNING = True
DISABLE = False


class timedRectAndEmoji:
    def __init__(self, rect, emoji, is_shown):
        self.emoji = emoji
        self.rect = rect
        self.time_when_drawn = time.time()
        self.is_show_on_screen = is_shown

    def update_time(self):
        self.time_when_drawn = time.time()


class App(tk.Tk):
    """
    emotion_detection: the class that detect the emotion in a face.
    score_threshold: the threshold for certainty level in the emotion detection, when decide if to draw the face on screen
    screen_number: not relevant right now.
    max_faces_to_draw: as it sound. max faces to show on screen.
    gesture_screen_time: how much time the gesture will be on screen. not very accurate.
    score_threshold_for_log: same as 'score_threshold' just for writing the face in the log files.
    """
    def __init__(self, emotion_detection: IEmotionDetection, score_threshold, screen_number=1, max_faces_to_draw=4, gesture_screen_time=5,
                 score_threshold_for_log=0.8):
        tk.Tk.__init__(self)
        self.floater = FloatingWindow(self, emotion_detection, score_threshold, screen_number, max_faces_to_draw,
                                      gesture_screen_time, score_threshold_for_log=score_threshold_for_log)
        self.withdraw()


""" 
this is the main class which handle the gui and the main logic.
in the `change_pic` function, it use 'NamesAndEmotionRecognition' to get the faces with the emotion,
then log it using 'ZELog' and draw the relevent faces on the screen.'
"""


class FloatingWindow(tk.Toplevel):
    def __init__(self, parent, emotion_detection: IEmotionDetection, score_threshold, screen_number=1, max_faces_to_draw=4,
                 gesture_screen_time=5, names_detection_period=5, score_threshold_for_log=0.8):
        super().__init__(parent)
        self.is_running = True
        self.time_delta_for_appearance = 4

        self.emotion_to_draw = [emotion.lower() for emotion in emotion_detection.emotion_to_draw]
        self.all_emotion = [emotion.lower() for emotion in emotion_detection.all_emotion]

        self.max_faces_to_draw = max_faces_to_draw
        self.gesture_screen_time = gesture_screen_time
        self.set_delete_counter_max(self.gesture_screen_time)
        self.screen_number = screen_number
        self.score_threshold = score_threshold
        self.parent = parent
        self.overrideredirect(True)
        self.wm_attributes("-topmost", True)
        self.wm_attributes("-transparentcolor", 'white')
        self.attributes('-alpha', 1)
        self.overrideredirect(True)
        self.overrideredirect(False)
        self.attributes('-fullscreen', True)
        self.size_change = tk.BooleanVar()

        # menu config
        home_path = path.normpath("menu-bar-smaller.png")
        self.home_icon = ImageTk.PhotoImage(Image.open(home_path))

        style = ttk.Style()
        style.configure("test.TButton", borderwidth=0)

        self.mb = ttk.Menubutton(self, text="Menu", style="test.TButton",
                                 image=self.home_icon)

        self.mb.pack(side="top", anchor="nw")

        self.menu = tk.Menu(self.mb, tearoff=False)

        self.menu.add_command(label='Disable', command=self.disable_btn)

        self.menu.add_command(label='Enable', command=self.enable_btn)
        self.menu.entryconfig("Enable", state="disabled")
        self.emotion_threshold_btn = self.menu.add_command(label="Emotions Threshold",
                                                           command=self.menu_emotion_threshold_btn)
        self.max_faces_to_draw_btn = self.menu.add_command(label="Max faces to draw",
                                                           command=self.menu_max_faces_to_draw_btn)
        self.gesture_screen_time_btn = self.menu.add_command(label="Gesture on screen time",
                                                             command=self.menu_gesture_screen_time_btn)

        self.menu.add_separator()
        self.menu.add_command(label='Exit', command=self.end_app)

        self.mb['menu'] = self.menu

        self.main_canvas = tk.Canvas(self, bg='white', highlightthickness=0)
        width = GetSystemMetrics(0)
        height = GetSystemMetrics(1)
        self.main_canvas.config(width=width, height=height)
        self.main_canvas.pack(anchor='nw', side='top')

        self.var = tk.IntVar()
        self.var.set(1)
        self.timer = Timer()
        self.timer.start()
        # self.emotion_recognition = er.EmotionDetection()
        self.name_and_emotion_recognition = NamesAndEmotionRecognition(self.screen_number, emotion_detection,
                                                                       num_threads=4)
        self.log = ZELog(score_threshold_for_log, self.all_emotion,
                         self.time_delta_for_appearance, self)
        self.faceRects = []
        self.faceRectsEmoji = []
        self.emoji_list = []
        self.is_disable = False
        self.app_running = True
        self.update()
        self.num_rects = 0
        self.all_gestures_objects = []
        self.used_gestures = []
        self.emoji_imgs = dict(zip(self.emotion_to_draw, [self.get_emoji(emotion) for emotion in self.emotion_to_draw]))

        self.cnt_for_avrege_time = 0
        self.avrege = 0
        self.emotion_threshold_pop = None
        self.max_faces_to_draw_pop = None
        self.gesture_screen_time_pop = None
        self.names_detection_period = names_detection_period

    def set_score_threshold(self, score_threshold):
        self.score_threshold = score_threshold
        self.name_and_emotion_recognition.set_score_threshold(score_threshold)
        self.log.set_score_threshold(score_threshold)

    def menu_emotion_threshold_btn(self):

        if self.emotion_threshold_pop is not None and self.emotion_threshold_pop.winfo_exists():
            self.emotion_threshold_pop.focus()
            return
        self.emotion_threshold_pop = tk.Toplevel(self)
        self.emotion_threshold_pop.title("Set Emotions Threshold")
        self.emotion_threshold_pop.geometry("250x180")
        self.emotion_threshold_pop.resizable(0, 0)

        var = tk.DoubleVar()
        var.set(int(self.score_threshold * 100))

        def select():
            set_button.config(relief=tk.SUNKEN)
            set_button.after(100, lambda: set_button.config(relief=tk.RAISED))
            threshold_value_in_percentage = var.get()
            selection = "Threshold Value = " + str(int(threshold_value_in_percentage)) + "%"
            new_threshold_label.config(text=selection)
            self.set_score_threshold(threshold_value_in_percentage / 100)

        explanation_label = tk.Label(self.emotion_threshold_pop,
                                     text="Only emotions with score\n above the value selected,\n will be displayed.")
        seperateor = ttk.Separator(self.emotion_threshold_pop, orient=tk.HORIZONTAL)
        threshold_scale = tk.Scale(self.emotion_threshold_pop, variable=var, from_=1, to=100, resolution=1,
                                   orient=tk.HORIZONTAL)
        set_button = tk.Button(self.emotion_threshold_pop, text="Set Threshold Value", command=select)
        new_threshold_label = tk.Label(self.emotion_threshold_pop)
        new_threshold_label.config(text="Threshold Value = " + str(int(self.score_threshold * 100)) + "%")

        explanation_label.pack(anchor=tk.CENTER)
        # seperateor.place(relx=0, rely=0.3, relwidth=2, relheight=1)
        seperateor.pack(anchor=tk.CENTER, fill='x', pady=5)
        threshold_scale.pack(anchor=tk.CENTER)
        set_button.pack(anchor=tk.CENTER)
        new_threshold_label.pack(anchor=tk.CENTER)

        threshold_scale.config()

    def menu_max_faces_to_draw_btn(self):
        if self.max_faces_to_draw_pop is not None and self.max_faces_to_draw_pop.winfo_exists():
            self.max_faces_to_draw_pop.focus()
            return
        self.max_faces_to_draw_pop = tk.Toplevel(self)
        self.max_faces_to_draw_pop.title("Set max faces to draw.")
        self.max_faces_to_draw_pop.geometry("180x140")
        self.max_faces_to_draw_pop.resizable(0, 0)
        var = tk.DoubleVar()
        var.set(self.max_faces_to_draw)
        max_faces_scale = tk.Scale(self.max_faces_to_draw_pop, variable=var, from_=1, to=20, resolution=1,
                                   orient=tk.HORIZONTAL)
        max_faces_scale.pack(anchor=tk.CENTER)

        def select():
            set_button.config(relief=tk.SUNKEN)
            set_button.after(100, lambda: set_button.config(relief=tk.RAISED))
            max_faces_to_draw = int(var.get())
            self.set_max_faces_to_draw(max_faces_to_draw)

        set_button = tk.Button(self.max_faces_to_draw_pop, text="Set max faces to draw", command=select)
        set_button.pack(anchor=tk.CENTER)

    def menu_gesture_screen_time_btn(self):
        if self.gesture_screen_time_pop is not None and self.gesture_screen_time_pop.winfo_exists():
            self.gesture_screen_time_pop.focus()
            return
        self.gesture_screen_time_pop = tk.Toplevel(self)
        self.gesture_screen_time_pop.title("Set max faces to draw.")
        self.gesture_screen_time_pop.geometry("180x140")
        self.gesture_screen_time_pop.resizable(0, 0)
        var = tk.DoubleVar()
        var.set(self.gesture_screen_time)
        gesture_screen_time_scale = tk.Scale(self.gesture_screen_time_pop, variable=var, from_=1, to=10, resolution=1,
                                             orient=tk.HORIZONTAL)
        gesture_screen_time_scale.pack(anchor=tk.CENTER)

        def select():
            set_button.config(relief=tk.SUNKEN)
            set_button.after(100, lambda: set_button.config(relief=tk.RAISED))
            gesture_screen_time = int(var.get())
            self.set_delete_counter_max(gesture_screen_time)

        set_button = tk.Button(self.gesture_screen_time_pop, text="Set gesture screen time \nin seconds",
                               command=select)
        set_button.pack(anchor=tk.CENTER)

    def disable(self):
        self.var.set(0)
        self.timer.pause()
        self.is_disable = True

    def disable_btn(self):
        self.menu.entryconfig("Disable", state="disabled")
        self.menu.entryconfig("Enable", state="normal")
        self.disable()

    def enable(self):
        self.var.set(1)
        self.timer.continue_()
        self.is_disable = False

    def enable_btn(self):
        self.menu.entryconfig("Enable", state="disabled")
        self.menu.entryconfig("Disable", state="normal")
        self.enable()

    def end_app(self, log_data=True):
        self.is_running = False
        self.parent.quit()
        self.destroy()
        self.timer.stop()
        if log_data:
            self.log.process_log_data(self.timer.get_total_time())
        self.name_and_emotion_recognition.destroy()

    def face_already_shown(self, face):

        rects_overlap = self.main_canvas.find_overlapping(face.left, face.down, face.right, face.up)
        return rects_overlap

    def move_gesture_to_current_face(self, face, old_gesture: timedRectAndEmoji):
        old_gesture.update_time()
        rect = old_gesture.rect
        emoji = old_gesture.emoji
        self.main_canvas.coords(rect, face.left, face.down, face.right, face.up)
        self.main_canvas.itemconfig(emoji, image=self.emoji_imgs[face.emotion])
        self.main_canvas.coords(emoji, face.left, face.down)
        self.set_gesture_as_shown_on_screen(old_gesture)

    def add_gestures(self, num_rects_to_add):
        for _ in range(num_rects_to_add):
            rect = self.main_canvas.create_rectangle(0, 0, 0, 0, outline="maroon", width=3)
            emoji = self.main_canvas.create_image(0, 0, image=self.emoji_imgs[self.emotion_to_draw[0]], anchor='se')
            gesture = timedRectAndEmoji(rect, emoji, is_shown=False)
            self.all_gestures_objects.append(gesture)

    @staticmethod
    def get_emoji(emotion):
        suffix = "png"
        emoji_path = f'emoji\{emotion.lower()}.{suffix}'
        emoji = Image.open(emoji_path)
        emoji = ImageTk.PhotoImage(emoji)
        return emoji

    def remove_gestures_from_screen(self, force=False):

        to_remove = []
        for i in range(len(self.used_gestures)):
            gesture = self.used_gestures[i]
            if time.time() - gesture.time_when_drawn >= self.gesture_screen_time or force:
                gesture.update_time()
                self.main_canvas.coords(gesture.rect, 0, 0, 0, 0)
                self.main_canvas.coords(gesture.emoji, 0, 0)
                self.set_gesture_as_not_shown_on_screen(gesture)
                self.all_gestures_objects.append(gesture)
                to_remove.append(gesture)
            else:
                self.used_gestures[i].time_when_drawn += 1
        self.used_gestures = [gesture for gesture in self.used_gestures if gesture not in to_remove]

    # main function
    def change_pic(self):
        after_time = 100
        if self.is_disable:
            self.disable_gui()
        faces = []
        self.remove_gestures_from_screen()
        # executor = ProcessPoolExecutor(max_workers=1)
        # faces_proc = executor.submit(self.name_and_emotion_recognition.detect_faces_from_screen())
        # print(faces_proc)
        # faces = faces_proc.result
        # print(faces)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.name_and_emotion_recognition.detect_faces_from_screen)
            faces = future.result()
        #faces = self.name_and_emotion_recognition.detect_faces_from_screen()
        if len(faces) > 0:
            self.log.log_faces(faces)
            faces_to_draw = self.pick_faces_to_draw(faces)
            # faces_to_draw = []
            self.create_gestures_for_faces(faces_to_draw)

            # moving existing gestures to the current faces
            while len(faces_to_draw) > 0:
                old_gesture = self.all_gestures_objects.pop()
                face = faces_to_draw.pop()
                if not self.face_already_shown(face):
                    self.move_gesture_to_current_face(face, old_gesture)
                    self.used_gestures.append(old_gesture)
                    self.update()
        if self.is_disable:
            self.disable_gui()
        elif not self.is_running:
            return
        self.after(after_time, self.change_pic)

    def set_max_faces_to_draw(self, max_faces_to_draw):
        self.max_faces_to_draw = max_faces_to_draw

    def set_delete_counter_max(self, gesture_screen_time):
        self.gesture_screen_time = gesture_screen_time

    def is_emotion_to_draw(self, emotion):
        return emotion in self.emotion_to_draw

    def get_num_faces_on_screen(self):
        return sum(1 if gesture.is_show_on_screen else 0 for gesture in self.used_gestures)

    @staticmethod
    def set_gesture_as_shown_on_screen(gesture):
        gesture.is_show_on_screen = True

    @staticmethod
    def set_gesture_as_not_shown_on_screen(gesture):
        gesture.is_show_on_screen = False

    def pick_faces_to_draw(self, faces):
        faces_to_draw = []
        for face in faces:
            if self.is_emotion_to_draw(face.emotion) and face.score >= self.score_threshold:
                print(face.score)
                faces_to_draw.append(face)

        faces_to_draw = self.take_faces_until_reach_max_faces_on_screen(faces_to_draw)
        return faces_to_draw

    def take_faces_until_reach_max_faces_on_screen(self, faces_to_draw):
        faces_on_screen = self.get_num_faces_on_screen()
        num_faces_to_draw = self.max_faces_to_draw - faces_on_screen
        return self.get_best_n_faces(faces_to_draw, num_faces_to_draw)

    @staticmethod
    def get_best_n_faces(faces_to_draw, num_faces_to_draw):
        if len(faces_to_draw) == 0:
            return []
        faces_to_draw = list(sorted(faces_to_draw, key=lambda face: face.score))
        return faces_to_draw[:num_faces_to_draw]

    def create_gestures_for_faces(self, faces_to_draw):
        if len(faces_to_draw) > len(self.all_gestures_objects):
            self.add_gestures(len(faces_to_draw) - len(self.all_gestures_objects))

    def disable_gui(self):
        self.remove_gestures_from_screen(True)
        self.mb.wait_variable(app.floater.var)


if __name__ == '__main__':
    score_threshold = 0.15
    emotions_detection = RonNetEmotionDetection()
    app = App(emotions_detection, score_threshold=score_threshold, screen_number=1, max_faces_to_draw=4,
              score_threshold_for_log=score_threshold)
    curr_time = time.time()
    if app.floater.is_running:
        app.floater.change_pic()
        app.floater.mainloop()
