from __future__ import division, print_function, absolute_import
from sys import maxsize

from timeit import time
import warnings
import cv2
from mot_woker import MOTWorker

import threading
from PIL import Image
from yolo import YOLO

from utils.video_shower import VideoShower

from multiprocessing import Process, Value, Queue

warnings.filterwarnings('ignore')

model = YOLO()
input_playback_f_q = Queue()
input_mot_f_q = Queue()
output_bbox_q = Queue()
output_playback_q = Queue()



def frameGetter():
    video_filename = -1
    video_capture = cv2.VideoCapture(video_filename)
    global input_playback_f_q
    global input_mot_f_q

    while True:
        ok, frame = video_capture.read()  
        if ok != True:
            break;

        if (input_mot_f_q.qsize() <= 1):
            input_mot_f_q.put(frame)
            # print('@@ input_mot_f_q qsize: ', input_mot_f_q.qsize())

        if (input_playback_f_q.qsize() <= 1):
            input_playback_f_q.put(frame)
            # print('@@ input_playback_f_q qsize: ', input_playback_f_q.qsize())
    video_capture.release()

def playbackVideo():
    global input_playback_f_q
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

    while True:
        if input_playback_f_q.empty() == True:
            continue;

        frame = input_playback_f_q.get()
        # print('@@ playback_thread frame: ', frame)
        cv2.imshow(win_name, frame)
        cv2.resizeWindow(win_name, 960, 540)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def detector2(frame):
    global model
    if frame is not None:
        image = Image.fromarray(frame)
        # if (model == None):
        #     model = YOLO()
        t1 = time.time()
        dets = model.detect_image(image)
        print('@@ Det time: ', time.time() - t1)
        if (len(dets) > 0):
            print('@@ Dets: ', dets[0])

def main():
    frame_getter_thread = threading.Thread(target=frameGetter, args=())
    playback_thread = threading.Thread(target=playbackVideo, args=())

    frame_getter_thread.start()
    # frame_getter_thread.join()
    playback_thread.start()

    # global input_mot_f_q

    while True:
        if input_mot_f_q.empty() == True:
            continue;

        frame = input_mot_f_q.get()
        detector2(frame)

     # for i in range(1):
        #    model = YOLO()
        #    thr = threading.Thread(target=detector2, args=(frame, model))
        #    thr.start()
        #    thr.join()   

        # if dets is not None:
        #     print('@@ dets: ', dets)

if __name__ == '__main__':
    main() 
