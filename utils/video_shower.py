import random
import cv2
import numpy as np 

class VideoShower ():
  # def __init__ (self, path, fps):
  #   self.fps = fps
  #   self.path = path
  def start (self, showerVideo, showerFrameQueue, showerBBQueue, fpsQueue):
    print('===== \nStarting showerVideo: {0} \n showerFrameQueue: {1} \n showerBBQueue: {2}\n'.format(showerVideo.value, showerFrameQueue.qsize(), showerBBQueue.qsize()))
    oldBBox = None
    oldFPS = None
    win_name = "output"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    try:
      while showerVideo.value or not showerFrameQueue.empty():
        #if bbox queue is not empty then get bbox and write it. Otherwise do not nothing.
        # print('@@ showerFrameQueue.qsize: ', showerFrameQueue.qsize())
        # print('@@ showerBBQueue.qsize: ', showerBBQueue.qsize())
        if not showerFrameQueue.empty():
          # print('@@ showerFrameQueue be qsize: ', showerFrameQueue.qsize())
          frame = showerFrameQueue.get()
          # print('@@ showerFrameQueue af qsize: ', showerFrameQueue.qsize())

          if not showerBBQueue.empty():
            (xmin, ymin, boxw, boxh) = showerBBQueue.get()
            # print('@@ shower Prc bbox: ', xmin, ymin, boxw, boxh)
            cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,255), 2) # BGR color
            # cv2.putText(frame, str(j),(xmin,ymin),0, 5e-3 * 200, (0,255,0),2)
            oldBBox = (xmin, ymin, boxw, boxh)
            
          else:
            if (oldBBox is not None):
              (xmin, ymin, boxw, boxh) = oldBBox
              # print('@@ shower Prc bbox: ', xmin, ymin, boxw, boxh)
              cv2.rectangle(frame, (xmin,ymin), (xmin+boxw,ymin+boxh), (255,0,255), 2) # BGR color

          font = cv2.FONT_HERSHEY_SIMPLEX
          if not fpsQueue.empty():
            fps = fpsQueue.get()
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
    except Exception as e:
      raise e
    