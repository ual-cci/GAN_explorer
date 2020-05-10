import cv2
import numpy as np

class Plotter(object):
    """
    Responsible for plotting experiments results
    """

    def __init__(self, renderer, getter):
        self.renderer = renderer
        self.getter = getter




    def plot(self, current_point):
        print("plotter called!")

        latents = np.asarray([current_point])
        image = self.getter.latent_to_image_localServerSwitch(latents)

