from multiprocessing import Queue
import random
import cv2
import numpy as np 

bkBBoxQ = Queue()
class VideoShower ():
  def start (self, showerVideo, showerFrameQueue, showerBBQueue, fpsQueue):
    global bkBBoxQ
    bkBBoxQ = Queue()
    oldFPS = None
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    try:
      while showerVideo.value or not showerFrameQueue.empty():
        #if bbox queue is not empty then get bbox and write it. Otherwise do not nothing.
        if not showerFrameQueue.empty():
          frame = showerFrameQueue.get()
          
          # draw Bounding box
          if not showerBBQueue.empty():
            # print('@@ bkBBoxQ bf size: ', bkBBoxQ.qsize())
            for i in range(showerBBQueue.qsize()):
                bboxsInF = showerBBQueue.get()
                bkBBoxQ.put(bboxsInF)
                for j in bboxsInF:
                  (xmin, ymin, boxw, boxh) = j
                  cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (50,255,255), 2) # BGR color
                  # bkBBoxQ.put_nowait((xmin, ymin, boxw, boxh)) # put new BBox to Backup queue
          else:
            if (bkBBoxQ.qsize() > 0 ): 
              # print('@@ 1 bkBBoxQ size: ', bkBBoxQ.qsize())
              for ii in range(bkBBoxQ.qsize()):
                bboxsInF = bkBBoxQ.get()
                for jj in bboxsInF:
                  (xmin, ymin, boxw, boxh) = jj
                  cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (50,255,255), 2) # BGR color
                  # print('@@ 3 draw old BBoxs ', xmin, ymin, boxw, boxh)
          
          # draw FPS
          font = cv2.FONT_HERSHEY_SIMPLEX
          if not fpsQueue.empty():
            fps = fpsQueue.get()
            cv2.putText(frame, fps, (10, 50), font, 1, (50,255,255), 2)
            oldFPS = fps
          else:
            cv2.putText(frame, oldFPS, (10, 50), font, 1, (50,255,255), 2)

          cv2.imshow(win_name, frame)
          cv2.resizeWindow(win_name, 960, 540)

          # Press Q to stop!
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          
      cv2.destroyAllWindows()
    except Exception as e:
      raise e
    