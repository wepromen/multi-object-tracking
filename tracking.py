#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import threading
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from utils.video_shower import VideoShower
from yolo import YOLO

from multiprocessing import Process, Value, Queue
from utils.video_writer import VideoWriter

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

#from optical_flow.optical import optical_flow_tracking
from optical_flow.getFeatures import getFeatures
from optical_flow.estimateAllTranslation import estimateAllTranslation
from optical_flow.applyGeometricTransformation import applyGeometricTransformation

yolo = YOLO()

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

def mot_task(frame, showerBBQueue, fpsQueue):
    t1 = time.time()

    print('@@ mot_task: 1')
    # yolo = YOLO()
    print('@@ mot_task: 2 ', type(yolo))

    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap1 = 1.0
    
    print('@@ mot_task: ', 1)


    # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    fps = 0.0

    image = Image.fromarray(frame)

    print('@@ mot_task: ', image)

    ## yolo detection
    boxs = yolo.detect_image(image) # [x,y,w,h]
    print("\n @@ box_num",len(boxs))
    features = encoder(frame,boxs)
    
    # score to 1.0 here).
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
    
    # Run non-maxima suppression (NMS)
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
    detections = [detections[i] for i in indices]

    ### Call the tracker
    tracker.predict()
    tracker.update(detections)
    

    ### Add one more step of optical flow
    # convert detections to bboxs for optical flow
    n_object = len(detections)
    bboxs = np.empty((n_object,4,2), dtype=float)
    i = 0
    for det in detections:
        bbox = det.to_tlbr() # (min x, min y, max x, max y)
        (xmin, ymin, boxw, boxh) = (int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1]))
        bboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
        i = i+1

    if firstflag:
        oldframe = frame
    else:
        startXs,startYs = getFeatures(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),bboxs,use_shi=False)
        newXs, newYs = estimateAllTranslation(startXs, startYs, oldframe, frame)
        Xs, Ys, newbboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)
        oldframe = frame
        ## generate new detections
        boxs = bbox_transform(newbboxs)
        features = encoder(frame,boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
        detections = [detections[i] for i in indices]

        ## Call the tracker again
        tracker.predict()
        tracker.update(detections)

    boxes_tracking = np.array([track.to_tlwh() for track in tracker.tracks])
    ### Deep sort tracker visualization
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        bboxTub = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        print('@@ Main Prc put showerBBQueu: ', bboxTub)
        showerBBQueue.put(bboxTub)
        print('@@ Main Prc put showerBBQueu: ', showerBBQueue.qsize())
        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(51,0,102), 2)
        # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2) 

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
        
    fps = 0.0
    firstflag = 1
    while True:
        ok, frame = video_capture.read()  # frame shape 640*480*3
        if ok != True:
            break;

        if videoShowerProcess is not None:
            showerFrameQueue.put(frame)
            print('@@ === Put frame to shower Proc')

        ## ================== Start MOT thread
        # import concurrent.futures as con
        # with con.ThreadPoolExecutor(max_workers=24) as excuter:
        #     excuter.map(mot_task(frame, yolo, encoder, nms_max_overlap1, tracker, OPTICAL, firstflag, fps), range(24))
        # args = (frame, yolo, encoder, nms_max_overlap1, tracker, OPTICAL, firstflag, showerBBQueue, fpsQueue)
        worker1 = threading.Thread(target= mot_task, args=(frame, showerBBQueue, fpsQueue,))
        worker1.start()
        ## ================== Start MOT thread


        # t1 = time.time()

        # image = Image.fromarray(frame)
        
        # ## yolo detection
        # boxs = yolo.detect_image(image) # [x,y,w,h]
        # # print("\n @@ box_num",len(boxs))
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


        # ### Start from the first frame, do optical flow for every two consecutive frames.
        # if OPTICAL:
        #     if firstflag:
        #         n_object = len(detections)
        #         bboxs = np.empty((n_object,4,2), dtype=float)
        #         i = 0
        #         for det in detections:
        #             bbox = det.to_tlbr() # (min x, min y, max x, max y)
        #             (xmin, ymin, boxw, boxh) = (int(bbox[0]), int(bbox[1]), int(bbox[2])-int(bbox[0]), int(bbox[3])-int(bbox[1]))
        #             bboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
        #             i = i+1
        #         startXs,startYs = getFeatures(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),bboxs,use_shi=False)
        #         oldframe = frame
        #         oldbboxs = bboxs
        #     else:
        #         ### add new tracking object
        #         # new_n_object = len(detections)
        #         # if new_n_object > n_object:
        #         #     # Run non-maxima suppression (NMS)
        #         #     tmp_boxes = np.array([d.tlwh for d in detections])
        #         #     tmp_scores = np.array([d.confidence for d in detections])
        #         #     tmp_indices = preprocessing.non_max_suppression(tmp_boxes, nms_max_overlap2, tmp_scores)
        #         #     tmp_detections = [detections[i] for i in indices]
        #         # if len(tmp_detections)>n_object:

        #         newXs, newYs = estimateAllTranslation(startXs, startYs, oldframe, frame)
        #         Xs, Ys, newbboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, oldbboxs)
        #         # update coordinates
        #         (startXs, startYs) = (Xs, Ys)

        #         oldframe = frame
        #         oldbboxs = newbboxs

        #         # update feature points as required
        #         n_features_left = np.sum(Xs!=-1)
        #         print('# of Features: %d'%n_features_left)
        #         if n_features_left < 15:
        #             print('Generate New Features')
        #             startXs,startYs = getFeatures(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),newbboxs)

        #         # draw bounding box and visualize feature point for each object
        #         for j in range(n_object):
        #             (xmin, ymin, boxw, boxh) = cv2.boundingRect(newbboxs[j,:,:].astype(int))
        #             # bboxQItem = {"xmin": xmin, "ymin": ymin, "boxw": boxw, "boxh": boxh}
        #             print('@@ Main Pr put showerBBQueu: ', xmin, ymin, boxw, boxh)
        #             showerBBQueue.put((xmin, ymin, boxw, boxh))
        #             print('@@ Main Pr put showerBBQueu: ', showerBBQueue.qsize())
        #             # cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,255,255), 2) # BGR color
        #             # cv2.putText(frame, str(j),(xmin,ymin),0, 5e-3 * 200, (0,255,0),2)
        #             # red color features
        #             # for k in range(startXs.shape[0]):
        #             #     cv2.circle(frame, (int(startXs[k,j]),int(startYs[k,j])),3,(0,0,255),thickness=2)

        # for det in detections:
        #     bbox = det.to_tlbr()
            # cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,51,103), 2) # BGR color
            
        # cv2.imshow('', frame)

        firstflag = 0
            
        # fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # print("fps= %f"%(fps))
        
        # textFPS = 'FPS: {:.2f}'.format(fps)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        
        # cv2.putText(frame, textFPS, (10, 20), font, 0.5, (255, 0, 255), 2)

        # cv2.imshow('', frame)
        


        if writeVideo_flag:
            # save a frame
            # out.write(frame)
            # put frame in queue
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

        # Press Q to stop!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

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
    if worker1 is not None:
        worker1.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
    # main(YOLO()) 
