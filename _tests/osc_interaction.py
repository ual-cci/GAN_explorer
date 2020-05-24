# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html#trackbar
# https://github.com/kivy/oscpy

import cv2
import numpy as np
from oscpy.client import OSCClient

class OSCSender(object):
    """
    Sends OSC messages from GUI
    """

    def onChangeSend(self,x):

        percentage = cv2.getTrackbarPos('Percentage', self.window_name)
        reset_toggle = cv2.getTrackbarPos('Reset points', self.window_name)

        # TEST:
        latent_vector_size = 512
        signal_latent = np.random.randn(1, latent_vector_size)[0]
        signal_latent = [float(v) for v in signal_latent]

        print("Sending message=", [percentage, reset_toggle, len(signal_latent)])
        self.osc.send_message(b'/send_gan_i', [percentage, reset_toggle] + signal_latent)

    def __init__(self):
        #address = "127.0.0.1"
        #port = 8000
        address = '0.0.0.0'
        port = 8000
        self.osc = OSCClient(address, port)
        self.window_name = 'Control'

    def start_window_rendering(self):

        # Create a black image, a window
        h = 75
        img = np.zeros((h, 512, 3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (0, h - 25)
        fontScale = 1
        fontColor = (128, 128, 128)
        lineType = 2

        cv2.namedWindow('Control')

        # create trackbars for color change
        cv2.createTrackbar('Percentage', self.window_name, 0, 1000, self.onChangeSend)
        cv2.createTrackbar('Reset points', self.window_name, 0, 1, self.onChangeSend)

        while (1):
            # also keep another inf. loop

            cv2.imshow('Control', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # get current positions of four trackbars
            r = cv2.getTrackbarPos('Percentage', self.window_name)
            r = int(r)

            img[:] = [r, r, r]
            cv2.putText(img, 'Select value: (0 to 1000 => %)', position, font, fontScale, fontColor, lineType)

        cv2.destroyAllWindows()


from threading import Thread
# Maybe move this into client?
osc_handler = OSCSender()
thread = Thread(target=osc_handler.start_window_rendering())
thread.start()
