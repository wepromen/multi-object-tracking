from multiprocessing import Queue
import random
import time
import cv2
import numpy as np 

class VideoShower ():
  def start (self, showerVideo, showerFrameQueue, showerBBQueue, fpsQueue, videoFps):
    # video_fps = 0.0
    bkBBoxArr = []
    oldFPS = None
    win_name = "output"
    # frame_count = 0
    no_bb_count = 0
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    try:
      while showerVideo.value or not showerFrameQueue.empty():
        t1 = time.time()
        #if bbox queue is not empty then get bbox and write it. Otherwise do not nothing.
        if not showerFrameQueue.empty():
          # frame_count += 1
          # print("@@ Frame: ", frame_count)
          frame = showerFrameQueue.get()
          # draw Bounding box
          if not showerBBQueue.empty():
            # print('@@ Draw new BBox: ', showerBBQueue.qsize())
            # print('@@ bkBBoxArr bf size: ', bkBBoxArr.qsize())
            if (len(bkBBoxArr) > 0):
              bkBBoxArr.clear()
            for i in range(showerBBQueue.qsize()):
              (xmin, ymin, boxw, boxh) = showerBBQueue.get()
              cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (50,255,255), 2) # BGR color
              bkBBoxArr.append((xmin, ymin, boxw, boxh)) # put new BBox to Backup queue
          else:
            if (no_bb_count > 7):
                no_bb_count = 0
                bkBBoxArr.clear()  
            if (len(bkBBoxArr) > 0 ): # and no_bb_count <= 200
              no_bb_count += 1
              # print('@@ Draw old BBox: ', len(bkBBoxArr))
              for i in bkBBoxArr:
                (xmin, ymin, boxw, boxh) = i
                cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (50,255,255), 2) # BGR color
                # print('@@ 3 draw old BBoxs ', xmin, ymin, boxw, boxh)
          # draw FPS
          font = cv2.FONT_HERSHEY_SIMPLEX
          if not fpsQueue.empty():
            fps = fpsQueue.get()
            cv2.putText(frame, fps, (10, 50), font, 0.6, (50,255,255), 2)
            oldFPS = fps
          else:
            cv2.putText(frame, oldFPS, (10, 50), font, 0.6, (50,255,255), 2)
          
          # Compute Video FPS
          # t_end = time.time()
          # print('@@  End-time: ', t_end - t1)
          # video_fps  = ( video_fps + (1./(t_end-t1)) ) / 2
          # print("video_fps= %f"%(video_fps))
          text_video_fps = 'video fps: {:}'.format(videoFps)
          cv2.putText(frame, text_video_fps, (10, 20), font, 0.6, (70,255,255), 2)

          cv2.imshow(win_name, frame)
          cv2.resizeWindow(win_name, 960, 540)

          # Press Q to stop!
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
      cv2.destroyAllWindows()
    except Exception as e:
      raise e
    