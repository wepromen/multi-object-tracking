from multiprocessing import Process, Queue
import time
from PIL import Image
from yolo import YOLO

yolo = None

class MOTWorker(Process):
    def __init__ (self, name, input_queue: Queue, output_queue: Queue, fpsQueue: Queue):
        super(MOTWorker, self).__init__()
        print(f"========== MOTWorker {name} started =============")
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.fpsQueue = fpsQueue

    def run(self):
        fps = 0.0
        while True:
            if self.input_queue.empty():
                continue
            else:
                frame = self.input_queue.get_nowait()

                t1 = time.time()
                image = Image.fromarray(frame)

                global yolo
                if (yolo == None):
                    yolo = YOLO()
                t2 = time.time()

                boxs = yolo.detect_image(image) # [x,y,w,h]
                
                t_detec_end = time.time()
                print('@@ Yolo det-time: ', t_detec_end - t2)
                fps  = ( fps + (1./(t_detec_end-t1)) ) / 2
                print("fps= %f"%(fps))
                textFPS = 'MOT FPS: {:.2f}'.format(fps)
                self.fpsQueue.put(textFPS)

                # print(f"@@ MOTWorker - BBoxs Ouput: ", len(boxs))
                if len(boxs) > 0:
                    if isinstance(boxs[0], list):
                        for i in boxs:
                            self.output_queue.put(i)
                    else:
                        self.output_queue.put(boxs)

