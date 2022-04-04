import time
from multiprocessing import Process, Queue

import cv2
import numpy as np
from PIL import Image
from yolo import YOLO


VIDEO_SOURCE = 0
# VIDEO_SOURCE = '/home/minhnn/Downloads/test4_unicam_body.mp4'


def _streamer(cap, ac_queue, bbox_queue):
    # cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FPS, 1)
    while True:
        frame_got, frame = cap.read()
        if frame_got:
            print("Got frame in streamer")
            if not bbox_queue.empty():
                boxs = bbox_queue.get()
                print('Getting boxs in streamer')
                for box in boxs:
                    xmin, ymin, boxw, boxh = box
                    frame = cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh))
                break
            cv2.imshow('Video', frame)
        if ac_queue.empty():
            print('Put to accomplish queue in streamer')
            ac_queue.put(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 


def _detector(model, ac_queue, bbox_queue):
    while True:
        if ac_queue.empty():
            print('Continue accomplish in detector')
            continue
        else:
            frame = ac_queue.get()
            print(frame)
            image = Image.fromarray(frame)
            boxs = model.detect_image(image)
            # print('Got boxs: {}'.format(boxs))
            bbox_queue.put(boxs)
            break


def main():
    init = time.time()
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FPS, 1)
    model = YOLO()
    bbox_queue = Queue()  # bbox budget
    ac_queue = Queue()    # accomplish queue
    # streamer = Process(target=_streamer)
    # detector = Process(target=_detector)
    streamer = Process(target=_streamer, args=(cap, ac_queue, bbox_queue))
    detector = Process(target=_detector, args=(model, ac_queue, bbox_queue))
    streamer.start()
    detector.start()
    streamer.join()
    detector.join()

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
    # cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_V4L)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # depends on fourcc available camera
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    main()