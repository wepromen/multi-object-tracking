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
    video_filename = 1
    # video_filename = './samples/u_frontcam_cuted_853x480.mp4'
    video_capture = cv2.VideoCapture(video_filename)
    # resize frame resolution
    # width, height = 960, 540

    # video_capture.set(cv2.CAP_PROP_FPS, 30)
    # v4l2-ctl -d /dev/video1 --list-formats-ext to see support resolutions
    # ret = video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    # ret1 = video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    # print('@@ set resolution: ', ret, ret1)
    video_fps = video_capture.get(cv2.CAP_PROP_FPS)
    print('@@ cap fps: ', video_fps)
    print('@@ frame resolution: ', video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #initial shower
    showerVideo = Value('i', 1)
    global showerBBQueue
    global showerFrameQueue
    global fpsQueue
    global mot_worker_input_queue

    # Start MOT Process
    number_of_mot_workers = 1
    for i in range(number_of_mot_workers):
        mot_worker = MOTWorker(input_queue=mot_worker_input_queue,
                                output_queue=showerBBQueue, name= '@@ MOTWorker ' + str(i), fpsQueue = fpsQueue)
        mot_worker.daemon = True
        print(mot_worker)
        mot_worker.start()
    # Start playback Process
    videoShower = VideoShower()
    videoShowerProcess = Process(target=videoShower.start, args=(showerVideo, showerFrameQueue, showerBBQueue, fpsQueue, video_fps ))
    videoShowerProcess.start()

    while True:
        ok, frame = video_capture.read()  # frame shape 640*480*3
        if ok != True:
            break;

        # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        # print('@@ cv2.resize: ', width, height)

        if showerFrameQueue.qsize() <= 1:
            showerFrameQueue.put(frame)
            # print('@@ === Put frame to shower Proc')
        
        if mot_worker_input_queue.qsize() <= 1:
            mot_worker_input_queue.put_nowait(frame)
            # print('@@ === Put frame qsize: ', mot_worker_input_queue.qsize())
        
    video_capture.release()

    if videoShowerProcess is not None:
        showerVideo.value = 0
        videoShowerProcess.join()
    if mot_worker is not None:
        mot_worker.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
