import codecs
import time
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
import os
from difflib import get_close_matches


class ZELog:
    def __init__(self, score_threshold, emotions, emotion_time_delta, gui):
        self.gui = gui
        self.old_names = []
        self.given_students_names = self.get_students_names()
        self.match_by_given_name = self.given_students_names is not None
        self.name_match_threshold_for_given_names = 0.5

        self.emotion_time_delta = emotion_time_delta
        self.score_threshold_for_log = score_threshold
        self.students_data = dict()
        self.names_in_current_image = []
        self.missing_emotion = "unknown"
        self.emotions = emotions
        self.summary_log_file_name = "summary log.csv"
        self.detailed_log_file_name = "detailed log.csv"
        self.logs_dir = "logs"
        self.start_time = time.time()
        self.name_match_threshold = 0.3

        self.delete_last_logs()

        self.emotions_in_graph = self.emotions + [self.missing_emotion]
        self.map = dict(zip(self.emotions_in_graph, range(len(self.emotions_in_graph))))

    def delete_last_logs(self):
        if not os.path.exists(self.logs_dir):
            os.mkdir(self.logs_dir)
        try:
            for filename in os.listdir(self.logs_dir):
                os.remove(self.logs_dir + "/" + filename)
        except IOError as e:
            print(e)
            self.gui.end_app(False)

    def set_score_threshold(self, threshold):
        self.score_threshold_for_log = threshold

    def process_log_data(self, total_run_time):

        emotion_time_data, emotion_per_time, names = self.calculate_emotion_time_data(self.students_data)
        self.write_log_to_file(total_run_time, emotion_time_data, emotion_per_time, names)

    def calculate_emotion_time_data(self, students_data):
        students_emotion_time = dict()
        students_emotions_per_time = dict()
        all_names = set()
        for student_name, student_data in students_data.items():
            all_names.add(student_name)
            student_emotion_time = dict()
            for emotion in self.emotions:
                student_emotion_time[emotion] = 0
            for student_time, emotion in student_data:
                student_emotion_time[emotion] += self.emotion_time_delta
                if student_time not in students_emotions_per_time:
                    students_emotions_per_time[student_time] = dict()
                students_emotions_per_time[student_time][student_name] = emotion
            students_emotion_time[student_name] = student_emotion_time
        return students_emotion_time, students_emotions_per_time, all_names

    def fill_missing_faces(self, curr_time):
        pass
        """for name in self.students_data.keys():
            if name not in self.names_in_current_image:
                self.students_data[name].append((curr_time, "missing"))"""

    def log_faces(self, faces):
        curr_time = time.time() - self.start_time
        for face in faces:
            self.names_in_current_image.append(face.name)
            if face.score >= self.score_threshold_for_log:
                self.log_face(face, curr_time)

        # self.fill_missing_faces(curr_time)
        self.names_in_current_image = []

    def log_face(self, face, curr_time):
        name = self.remove_unvalid_chars(face.name)
        name = self.try_match_to_old_names(name)

        if name is not None and name != "":
            if name not in self.students_data:
                self.students_data[name] = []
            self.students_data[name].append((curr_time, face.emotion))

    def write_log_to_file(self, total_run_time, emotion_time_data, emotion_per_time, names):
        detailed_data = self.write_detailed_log(emotion_per_time, names)
        self.write_summarize_log(emotion_time_data, total_run_time)
        self.write_graphs(detailed_data)

    def write_summarize_log(self, emotion_time_data, total_run_time):
        title = ",".join(["Name", "Total"] + self.emotions)
        students_text_list = [title]
        for student_name, emotions_data in emotion_time_data.items():
            each_emotion_time = emotions_data.values()
            total_emotion_time_percent = sum(each_emotion_time) / float(total_run_time)
            each_emotion_time_percent = list(map(lambda x: x / total_run_time, each_emotion_time))
            each_emotion_time_percent_str = list(map(str, each_emotion_time_percent))
            student_text = ",".join([student_name, str(total_emotion_time_percent)] + each_emotion_time_percent_str)
            students_text_list.append(student_text)
        students_text = "\n".join(students_text_list)
        my_file = open(self.logs_dir + "/" + self.summary_log_file_name, "w")
        my_file.write(students_text)
        my_file.close()

    def write_detailed_log(self, emotion_per_time, names):
        data_arr = df(columns=names)
        for my_time, emotions_names in emotion_per_time.items():
            time_str = str(my_time)
            data_arr.append(pd.Series(name=my_time))
            for name in emotions_names.keys():
                data_arr.at[time_str, name] = emotions_names[name]
        data_arr.index = data_arr.index.astype(float)
        data_arr = data_arr.sort_index()
        data_arr.to_csv(self.logs_dir + "/" + self.detailed_log_file_name, encoding='utf-8-sig')
        return data_arr

    def my_map_func(self, emotion):
        return self.map[emotion]

    @staticmethod
    def remove_unvalid_chars(filename):
        invalid = '<>:"/\|?* '
        for char in invalid:
            filename = filename.replace(char, '')
        return filename

    def write_graphs(self, detailed_data):
        detailed_data = detailed_data.fillna(self.missing_emotion)
        my_yticks = self.emotions_in_graph
        for student_name in detailed_data.columns:
            plt.figure()
            if student_name == "":
                return
            student_data = detailed_data[student_name]
            time_line = range(len(student_data))
            my_xticks = np.array(range(len(time_line)))
            student_emotions_values = list(map(self.my_map_func, student_data))
            y = np.array(range(len(my_yticks)))
            plt.xticks(time_line, my_xticks)
            plt.yticks(y, my_yticks)
            plt.xlabel("time")
            plt.scatter(time_line, student_emotions_values, c='crimson')
            plt.savefig(self.logs_dir + "/" + student_name + ".png")

    def try_match_to_old_names(self, name):
        if self.match_by_given_name:
            close_matches = get_close_matches(name, self.given_students_names, n=1,
                                              cutoff=self.name_match_threshold_for_given_names)
        else:
            close_matches = get_close_matches(name, self.old_names, n=1,
                                              cutoff=self.name_match_threshold)
            if len(close_matches) != 1:
                self.old_names.append(name)

        if len(close_matches) == 1:
            return close_matches[0]
        else:
            return name

    def get_students_names(self):
        file_name = "names.txt"
        try:
            f = codecs.open(file_name, "r", "utf-8")
        except Exception as e:
            print(e)
            return None

        names = [name.replace("\r", "").replace("\n", "") for name in f.readlines()]
        f.close()
        if len(names) == 0:
            return None
        return names





if __name__ == "__main__":
    file = open("log.csv", "w")
    file.write("fsf")
    file.close()
