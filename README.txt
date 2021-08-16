app structure:
	-gui is the app "manager". it control the main flow and call the FaceDetection and the ZELog during the flow.
	- FaceDetection responsable for process the screenshot and return list of faces in it, with their names and their emotion,
	- ZELog recieve list of faces and store them with time of receiving them. when exit the app, it create the logs from this data.

to change emotion detection:
	the App class receive emotion_detection object which is from the type IEmotionDetection.
	currently we pass an object that inherit from IEmotionDetection, and implemented with RonNet.
	to use another net, simply create a class which inherit from IEmotionDetection, and pass to to the App class instead of the current object.
	if your net have difreent emotions than ours, add emoji for each emotion in the emoji folder. (the name of the emoji file need to be as the emotion in lower case  e.g. "happy.PNG")

parameters passed to the App class:

    emotion_detection: the class that detect the emotion in a face.
    score_threshold: the threshold for certainty level in the emotion detection, when decide if to draw the face on screen
    screen_number: not relevant right now.
    max_faces_to_draw: as it sound. max faces to show on screen.
    gesture_screen_time: how much time the gesture will be on screen. not very accurate.
    score_threshold_for_log: same as 'score_threshold' just for writing the face in the log files.