from __future__ import division, print_function, absolute_import
from sys import maxsize

from timeit import time
import warnings
import cv2
from mot_woker import MOTWorker

from utils.video_shower import VideoShower

from multiprocessing import Process, Value, Queue

warnings.filterwarnings('ignore')

mot_worker_input_queue = Queue()
showerBBQueue = Queue()
fpsQueue = Queue()
showerFrameQueue = Queue()


def bbox_transform(newbboxs):
    return_boxs = []
    for i in range(newbboxs.shape[0]):
        [x,y,w,h] = [newbboxs[i,0,0],newbboxs[i,0,1],newbboxs[i,3,0]-newbboxs[i,0,0],newbboxs[i,3,1]-newbboxs[i,0,1]]
        return_boxs.append([x,y,w,h])
    return return_boxs

def main():
    ## get video source
    video_filename = -1
    # video_filename = './samples/u_frontcam_cuted_853x480.mp4'
    video_capture = cv2.VideoCapture(video_filename)

    #initial shower
    showerVideo = Value('i', 1)
    # showerBBQueue = Queue()
    global showerBBQueue
    global showerFrameQueue
    global fpsQueue
    global mot_worker_input_queue
    # showerFrameQueue = Queue()
    # fpsQueue = Queue()
    videoShower = VideoShower()
    videoShowerProcess = Process(target=videoShower.start, args=(showerVideo, showerFrameQueue, showerBBQueue, fpsQueue))
    videoShowerProcess.start()

    # NOTE: MOT Workers
    # mot_worker_input_queue = Queue()
    number_of_mot_workers = 1

    for i in range(number_of_mot_workers):
        mot_worker = MOTWorker(input_queue=mot_worker_input_queue,
                                output_queue=showerBBQueue, name= '@@ MOTWorker ' + str(i), fpsQueue = fpsQueue)
        mot_worker.daemon = True
        print(mot_worker)
        mot_worker.start()


    while True:
        ok, frame = video_capture.read()  # frame shape 640*480*3
        if ok != True:
            break;

        if videoShowerProcess is not None:
            showerFrameQueue.put(frame)
            # print('@@ === Put frame to shower Proc')
        # mot_worker_input_queue = Queue()
        if mot_worker_input_queue.qsize() == 0:
            print('@@ === Put frame: ', )
            mot_worker_input_queue.put(frame)
        
    # frame_rate = 10000
    # prev = 0

    # while True:
    #     time_elapsed = time.time() - prev
    #     ok, frame = video_capture.read()  # frame shape 640*480*3
    #     if ok != True:
    #         break;

    #     if time_elapsed > 1./frame_rate:
    #         prev = time.time()

    #         if videoShowerProcess is not None:
    #             showerFrameQueue.put(frame)
    #             # print('@@ === Put frame to shower Proc')

    #         mot_worker_input_queue.put(frame)

    video_capture.release()

    if videoShowerProcess is not None:
        showerVideo.value = 0
        videoShowerProcess.join()
    if mot_worker is not None:
        mot_worker.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
