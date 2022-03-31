#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import threading
from timeit import time
import warnings
import sys
import cv2
from mot_woker import MOTWorker
# import numpy as np
# from PIL import Image
from utils.video_shower import VideoShower
# from yolo import YOLO

from multiprocessing import Process, Value, Queue
from utils.video_writer import VideoWriter

# from deep_sort import preprocessing
# from deep_sort import nn_matching
# from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
# from deep_sort.detection import Detection as ddet

# #from optical_flow.optical import optical_flow_tracking
# from optical_flow.getFeatures import getFeatures
# from optical_flow.estimateAllTranslation import estimateAllTranslation
# from optical_flow.applyGeometricTransformation import applyGeometricTransformation

warnings.filterwarnings('ignore')

# transform newbboxs of (n_object,4,2) np array s.t. return_boxs = bbox_transform(newbboxs)
# newbboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
# return_boxs = [], return_boxs.append([x,y,w,h])
def bbox_transform(newbboxs):
    return_boxs = []
    for i in range(newbboxs.shape[0]):
        [x,y,w,h] = [newbboxs[i,0,0],newbboxs[i,0,1],newbboxs[i,3,0]-newbboxs[i,0,0],newbboxs[i,3,1]-newbboxs[i,0,1]]
        return_boxs.append([x,y,w,h])
    return return_boxs

def mot_task(frame,showerBBQueue, fpsQueue):
    t1 = time.time()
    print('@@ mot_task: 1',)

    # from deep_sort import preprocessing
    # from deep_sort import nn_matching
    # from deep_sort.detection import Detection
    # from deep_sort.tracker import Tracker
    # from tools import generate_detections as gdet
    # from deep_sort.detection import Detection as ddet

    # import numpy as np
    from PIL import Image

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap1 = 1.0
    
    print('@@ mot_task: ', 2)


    # deep_sort 
    # model_filename = 'model_data/mars-small128.pb'
    # encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)

    fps = 0.0

    image = Image.fromarray(frame)

    print('@@ mot_task: 3', image)

    ## yolo detection
    yolo = YOLO()
    boxs = yolo.detect_image(image) # [x,y,w,h]
    print("\n @@ mot_task: 4 box_num",boxs[0])
    showerBBQueue.put(boxs[0][0],boxs[0][1], boxs[0][2], boxs[0][3])

    # features = encoder(frame,boxs)
    
    # # score to 1.0 here).
    # detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    
    # # Run non-maxima suppression (NMS)
    # boxes = np.array([d.tlwh for d in detections])
    # scores = np.array([d.confidence for d in detections])
    # indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
    # detections = [detections[i] for i in indices]

    # ### Call the tracker
    # tracker.predict()
    # tracker.update(detections)
    

    # ### Add one more step of optical flow
    # # convert detections to bboxs for optical flow
    # n_object = len(detections)
    # bboxs = np.empty((n_object,4,2), dtype=float)
    # i = 0
    # for det in detections:
    #     bbox = det.to_tlbr() # (min x, min y, max x, max y)
    #     (xmin, ymin, boxw, boxh) = (int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1]))
    #     bboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
    #     i = i+1

    # if firstflag:
    #     oldframe = frame
    # else:
    #     startXs,startYs = getFeatures(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),bboxs,use_shi=False)
    #     newXs, newYs = estimateAllTranslation(startXs, startYs, oldframe, frame)
    #     Xs, Ys, newbboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)
    #     oldframe = frame
    #     ## generate new detections
    #     boxs = bbox_transform(newbboxs)
    #     features = encoder(frame,boxs)
    #     detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
    #     boxes = np.array([d.tlwh for d in detections])
    #     scores = np.array([d.confidence for d in detections])
    #     indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
    #     detections = [detections[i] for i in indices]

    #     ## Call the tracker again
    #     tracker.predict()
    #     tracker.update(detections)

    # boxes_tracking = np.array([track.to_tlwh() for track in tracker.tracks])
    # ### Deep sort tracker visualization
    # for track in tracker.tracks:
    #     if not track.is_confirmed() or track.time_since_update > 1:
    #         continue 
    #     bbox = track.to_tlbr()
    #     bboxTub = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    #     print('@@ Main Prc put showerBBQueu: ', bboxTub)
    #     showerBBQueue.put(bboxTub)
    #     print('@@ Main Prc put showerBBQueu: ', showerBBQueue.qsize())
    #     # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(51,0,102), 2)
    #     # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2) 

    # firstflag = 0
            
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %f"%(fps))
    
    textFPS = 'FPS: {:.2f}'.format(fps)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    fpsQueue.put(textFPS)
    # cv2.putText(frame, textFPS, (10, 20), font, 0.5, (255, 0, 255), 2)

    # cv2.imshow('', frame)

# def main(yolo):
def main():
    
    # # Definition of the parameters
    # max_cosine_distance = 0.3
    # nn_budget = None
    # nms_max_overlap1 = 1.0
    
    # # deep_sort 
    # model_filename = 'model_data/mars-small128.pb'
    # encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)

    writeVideo_flag = False 
    OPTICAL = False
    
    ## get video source
    video_filename = 0
    # video_filename = './samples/u_frontcam_cuted_853x480.mp4'
    video_capture = cv2.VideoCapture(video_filename)

    #instatiate video writer
    videoWriter = VideoWriter('outputs/output.avi', video_capture.get(cv2.CAP_PROP_FPS))
    videoWriterProcess = None

    #initial shower
    showerVideo = Value('i', 1)
    showerBBQueue = Queue()
    showerFrameQueue = Queue()
    fpsQueue = Queue()
    videoShower = VideoShower()
    videoShowerProcess = Process(target=videoShower.start, args=(showerVideo, showerFrameQueue, showerBBQueue, fpsQueue))
    videoShowerProcess.start()

    # NOTE: MOT Workers
    mot_worker_input_queue = Queue()
    # mot_worker_output_queue = Queue()
    number_of_mot_workers = 1

    for i in range(number_of_mot_workers):
        mot_worker = MOTWorker(input_queue=mot_worker_input_queue,
                                output_queue=showerBBQueue, name= '@@ MOTWorker ' + str(i), fpsQueue = fpsQueue)
        mot_worker.start()

    fps = 0.0
    firstflag = 1

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out = cv2.VideoWriter('outputs/output.avi', fourcc, 15, (w, h))

        if videoWriterProcess is None:
            writeVideo = Value('i', 1)
            frameQueue = Queue()
            videoWriterProcess = Process(target=videoWriter.start, args=(writeVideo, frameQueue, w, h))
            videoWriterProcess.start()

        list_file = open('outputs/detection.txt', 'w')
        list_file2 = open('outputs/tracking.txt', 'w')
        frame_index = -1 

    while True:
        ok, frame = video_capture.read()  # frame shape 640*480*3
        if ok != True:
            break;

        if videoShowerProcess is not None:
            showerFrameQueue.put(frame)
            # print('@@ === Put frame to shower Proc')

        mot_worker_input_queue.put(frame)

        firstflag = 0

        if writeVideo_flag:
            # save a frame
            if videoWriterProcess is not None:
                frameQueue.put(frame)
            # detection
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            # tracking
            list_file2.write(str(frame_index)+' ')
            if len(boxes_tracking) != 0:
                for i in range(0,len(boxes_tracking)):
                    list_file2.write(str(boxes_tracking[i][0]) + ' '+str(boxes_tracking[i][1]) + ' '+str(boxes_tracking[i][2]) + ' '+str(boxes_tracking[i][3]) + ' ')
            list_file2.write('\n')

    video_capture.release()
    if writeVideo_flag:
        # out.release()
        if videoWriterProcess is not None:
            writeVideo.value = 0
        videoWriterProcess.join()

        list_file.close()

    if videoShowerProcess is not None:
        showerVideo.value = 0
        videoShowerProcess.join()
    # if worker1 is not None:
    #     worker1.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
    # main(YOLO()) 
