from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt

import cv2
import threading
import time

from .models import setup, get_data_info, data_generator_rt
        

def index(request):
    return render(request, "index.html")

@gzip.gzip_page
def realtime(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, "realtime.html")

def fileRequest(request):
    if request.method == "POST":
        video = request.POST.get('video')
        cam = VideoCamera(file_path='media/sample.mov')
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    return redirect("index")

class VideoCamera(object):
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.video = cv2.VideoCapture(0) if file_path is None else cv2.VideoCapture(file_path)        
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        if self.file_path is None:
            new_width = image.shape[0]
            new_height = image.shape[1]
        else:
            new_width = image.shape[0]//3
            new_height = image.shape[1]//3
        image = cv2.resize(image, (new_height, new_width))
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    # init model
    model, mp_drawing, mp_drawing_styles, mp_pose = setup()

    # init data info
    labels_text, labels = get_data_info()

    # init aug
    time0 = 0
    sequence = []
    sentence = ['']

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while camera.video.isOpened():
            image_org = camera.get_frame()
            
            # convert to numpy
            image_org = np.array(image_org)
            
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image_org, 1)
            
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            if results.pose_landmarks:
                keypoint = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten()
                keypoint = np.array_split(keypoint, 33)
                sequence.append(keypoint)
                sequence = sequence[-116:]

            if len(sequence) == 116:
                X_test_rt = data_generator_rt(sequence[-96:])
                res = model.predict([X_test_rt])[0]
                print(labels_text[np.argmax(res)])
                sentence.append(labels[np.argmax(res)])
                sequence.clear()  

            image = cv2.flip(image_org, 1)
            
            # Show fps
            time1 = time.time()
            fps = 1 / (time1 - time0)
            time0 = time1
            cv2.putText(image,'FPS:' + str(int(fps)), (3, 475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ''.join(sentence[-1:]), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            image = cv2.imencode('.jpg', image)[1]

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n\r\n')