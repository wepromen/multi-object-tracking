from re import M
import time
from multiprocessing import Process, Queue

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO


VIDEO_SOURCE = -1

model = None

def _streamer(cap):
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    while True:
        frame_got, frame = cap.read()
        if frame_got != True:
            break;

        print("Got frame in streamer")
        t1 = time.time()
        boxs = detector2(frame)
        print('@@ Dectection time: ', time.time() - t1)
        print("@@ Yolo det boxs: ", boxs)
        if len(boxs) > 0:
            for box in boxs:
                    (xmin, ymin, boxw, boxh) = box
                    frame = cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,255), 2)
        
        cv2.imshow(win_name, frame)
        cv2.resizeWindow(win_name, 960, 540)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    cap.release()
    cv2.destroyAllWindows()
 
def detector2(frame):
    global model
    if (model == None):
        model = YOLO()
    if frame is not None:
        image = Image.fromarray(frame)
        boxs = model.detect_image(image)
        return boxs
    else:
        return None

def main():
    init = time.time()
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
    # global model
    # l_model = YOLO()
    # model = l_model
    bbox_queue = Queue()  # bbox budget
    ac_queue = Queue()    # accomplish queue
    # streamer = Process(target=_streamer)
    # detector = Process(target=_detector)
    streamer = Process(target=_streamer, args=(cap,))
    # detector = Process(target=_detector, args=(model, ac_queue, bbox_queue))
    streamer.start()
    # detector.start()
    streamer.join()
    # detector.join()

    ## Single task
    # begin = time.time()
    # print('Load model time: {}'.format(begin - init))
    # boxs = model.detect_image(image)
    # detect_time = time.time()
    # print('Detect time: {}'.format(detect_time - begin))
    # frame = np.asarray(image)
    # for box in boxs:
    #     xmin, ymin, boxw, boxh = box
    #     print(box)
    #     frame = cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,255), 2)
    # cv2.imshow('graycsale image',frame)
    # print('Draw time: {}'.format(time.time() - detect_time))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()