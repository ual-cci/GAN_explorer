import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread

import time
import collections

class FPS:
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0

class Renderer(object):
    """
    Draw image to screen.
    """

    def __init__(self, show_fps = True):
        self.show_fps = show_fps
        self.counter = 0
        return None

    def show_frames(self, get_image_function):
        fps = FPS()
        self.counter = 0

        while (True):
            self.counter += 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            """
            if key == ord('8'):
                self.sample_every /= 2
            if key == ord('2'):
                self.sample_every *= 2
            """

            # get image
            frame = None

            time_start = timer()
            frame = get_image_function(self.counter)
            time_end = timer()
            print("timer:", (time_end-time_start))
            #fps_val = 1.0 / (time_end-time_start)
            fps_val = fps()
            # includes time for the open cv to render and the text
            # so without showing it (imshow), it would be 24-26fps, with it it's cca 20-21fps
            #print(fps_val)

            if self.show_fps:
                frame = cv2.putText(frame, "FPS "+'{:.2f}'.format(fps_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('frame', frame)



