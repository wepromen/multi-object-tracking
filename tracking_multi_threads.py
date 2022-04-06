from __future__ import division, print_function, absolute_import
from sys import maxsize

# from timeit import time
import time
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
output_fps_q = Queue()
output_playback_q = Queue()
oldBBox = None
oldFPS = None

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
            input_mot_f_q.put_nowait(frame)
            # print('@@ input_mot_f_q qsize: ', input_mot_f_q.qsize())

        if (input_playback_f_q.qsize() <= 1):
            input_playback_f_q.put_nowait(frame)
            # print('@@ input_playback_f_q qsize: ', input_playback_f_q.qsize())
    video_capture.release()

def playbackVideo():
    global input_playback_f_q
    global output_bbox_q
    global output_fps_q
    global oldBBox
    global oldFPS

    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions

    while True:
        if input_playback_f_q.empty() == True:
            continue;

        frame = input_playback_f_q.get_nowait()
        # print('@@ playback_thread frame: ', frame)

        if not output_bbox_q.empty():
            # print('@@ output_bbox_q: ', output_bbox_q.get())
            (xmin, ymin, boxw, boxh) = output_bbox_q.get()
            # print('@@ shower Prc bbox: ', xmin, ymin, boxw, boxh)
            cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,255), 2) # BGR color
            oldBBox = (xmin, ymin, boxw, boxh)
        else:
            if (oldBBox is not None):
                (xmin, ymin, boxw, boxh) = oldBBox
                # print('@@ shower Prc bbox: ', xmin, ymin, boxw, boxh)
                cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,255), 2) # BGR color

        font = cv2.FONT_HERSHEY_SIMPLEX
        if not output_fps_q.empty():
            fps = output_fps_q.get()
            cv2.putText(frame, fps, (10, 50), font, 2.0, (255, 0, 255), 2)
            oldFPS = fps
        else:
            cv2.putText(frame, oldFPS, (10, 50), font, 2.0, (255, 0, 255), 2)

        cv2.imshow(win_name, frame)
        cv2.resizeWindow(win_name, 960, 540)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def detector2(frame):
    global model
    global output_bbox_q
    if frame is not None:
        image = Image.fromarray(frame)
        t2 = time.time()
        dets = model.detect_image(image)
        print('@@ Det time: ', time.time() - t2)
        if (len(dets) > 0):
            if isinstance(dets[0], list):
                for i in dets:
                    if (output_bbox_q.qsize() <= 1):
                        output_bbox_q.put(i)
            # else:
            #     output_bbox_q.put(dets)
        #     print('@@ Dets: ', dets[0])

def main():
    frame_getter_thread = threading.Thread(target=frameGetter, args=())
    playback_thread = threading.Thread(target=playbackVideo, args=())

    frame_getter_thread.start()
    # frame_getter_thread.join()
    playback_thread.start()
    fps = 0.0
    global output_fps_q

    while True:
        if input_mot_f_q.empty() == True:
            continue;

        frame = input_mot_f_q.get()
        t1 = time.time()
        detector2(frame)
        t_detec_end = time.time()
        fps  = ( fps + (1./(t_detec_end-t1)) ) / 2
        print("fps= %f"%(fps))
        textFPS = 'FPS: {:.2f}'.format(fps)
        if (output_fps_q.qsize() <= 1):
            output_fps_q.put(textFPS)

if __name__ == '__main__':
    main() 
