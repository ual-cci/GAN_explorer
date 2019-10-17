import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread

import renderer
from getter_functions import latent_to_image_localServerSwitch

class Interaction_Handler(object):
    """
    Do all the interaction tricks here.
    """

    def __init__(self):
        self.renderer = renderer.Renderer()
        self.renderer.show_fps = True

        self.latent_vector_size = 512

    # v0 - pure random
    def get_random_image(self, counter):
        how_many = 1
        latents = np.random.randn(how_many, self.latent_vector_size)
        return latent_to_image_localServerSwitch(latents)

    def start_renderer_no_interaction(self):
        self.renderer.show_frames(self.get_random_image)

    # v0b - interpolate between two random points, no interaction still really
    def shuffle_random_points(self, steps = 120):
        how_many = 2
        latents = np.random.randn(how_many, self.latent_vector_size)

        self.p0 = latents[0]
        self.p1 = latents[1]
        self.step = 0
        self.steps = steps

        #self.step = 30

        self.calculate_p = lambda step: self.p0 + (1.0 - float((self.steps - step) / self.steps)) * (self.p1 - self.p0)
        self.p = self.calculate_p(step=0)

        #alpha = 1.0 - float((self.steps - self.step) / self.steps)
        #self.p = self.p0 + alpha * (self.p1 - self.p0)

    def get_interpolated_image(self, counter):
        # counter goes from 0 to inf
        self.step = counter % self.steps  # from 0 to self.steps

        if self.step == 0:
            self.shuffle_random_points(self.steps)

        self.p = self.calculate_p(step=self.step)

        latents = np.asarray([self.p])
        return latent_to_image_localServerSwitch(latents)

    def start_renderer_interpolation(self):
        self.renderer.show_frames(self.get_interpolated_image)

    # v1 - interpolate between two points, use OSC signal
    def get_interpolated_image_OSC_input(self, counter):
        # ignore counter
        global SIGNAL_interactive_i
        global SIGNAL_reset_toggle

        if SIGNAL_reset_toggle == 1:
            self.shuffle_random_points(self.steps)

        alpha = SIGNAL_interactive_i
        self.p = self.p0 + (alpha) * (self.p1 - self.p0)

        latents = np.asarray([self.p])
        return latent_to_image_localServerSwitch(latents)

    def start_renderer_interpolation_interact(self):
        self.renderer.show_frames(self.get_interpolated_image_OSC_input)


interaction_handler = Interaction_Handler()

version = "v0"

if version == "v0":
    interaction_handler.start_renderer_no_interaction()

elif version == "v0b":
    interaction_handler.shuffle_random_points(steps=20)
    interaction_handler.start_renderer_interpolation()

elif version == "v1":
    OSC_address = '0.0.0.0'
    OSC_port = 8000
    OSC_bind = b'/send_gan_i'

    SIGNAL_interactive_i = 0.0
    SIGNAL_reset_toggle = 0

    # OSC - Interactive listener
    def callback(*values):
        global SIGNAL_interactive_i
        global SIGNAL_reset_toggle
        print("OSC got values: {}".format(values))
        # [percentage, model_i, song_i]
        percentage, reset_toggle = values

        SIGNAL_interactive_i = float(percentage) / 1000.0  # 1000 = 100% = 1.0
        SIGNAL_reset_toggle = int(reset_toggle)

    print("Also starting a OSC listener at ", OSC_address, OSC_port, OSC_bind, "to listen for interactive signal (0-1000).")
    from oscpy.server import OSCThreadServer
    osc = OSCThreadServer()
    sock = osc.listen(address=OSC_address, port=OSC_port, default=True)
    osc.bind(OSC_bind, callback)

    interaction_handler.shuffle_random_points(steps=20)
    interaction_handler.start_renderer_interpolation_interact()
