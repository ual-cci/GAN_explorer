import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread

import renderer
from getter_functions import latent_to_image_localServerSwitch, get_vec_size_localServerSwitch

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

    # v2 - controlled using wsad
    def get_interpolated_image_key_input(self, counter, key_code, key_ord):
        # ignore counter
        # look at the key command
        #if key_ord is not -1:
        #    print("key pressed (code, ord)", key_code, key_ord)

        # Save & Load
        if key_ord is 225 or key_ord is 226:
            self.SHIFT = not self.SHIFT
            print("Saving ON?:", self.SHIFT)
        if key_ord is 233 or key_ord is 234:
            self.ALT = not self.ALT

        nums = [str(i) for i in list(range(0,9))]

        if self.SHIFT and key_code in nums:
            # SAVE on position
            save_to_i = int(key_code)
            print("saving to ", save_to_i)
            self.saved[save_to_i] = self.p0
        if not self.SHIFT and key_code in nums:
            # LOAD from position
            load_from_i = int(key_code)
            print("loading from ", load_from_i)
            if self.saved[load_from_i] is not None:
                self.p0 = self.saved[load_from_i]

        # Random jump
        if key_code == "r":
            self.previous = self.p0
            self.shuffle_random_points(self.steps)

        # One undo
        if key_code == "e":
            tmp = self.p0
            self.p0 = self.previous
            self.previous = tmp

        # small jump
        if key_code == " ":
            save_p0 = self.p0
            self.shuffle_random_points(self.steps)

            alpha = 0.02
            self.p0 = (1.0 - alpha) * save_p0 + alpha*self.p0

        # AD => selecting a feature (0 to self.latent_vector_size and then loop)
        if key_code == "a" or key_code == "d":
            direction = -1 # left
            if key_code == "d":
                direction = +1  # right
            self.selected_feature_i = self.selected_feature_i + direction

            if self.selected_feature_i < 0:
                self.selected_feature_i = self.latent_vector_size - 1
            if self.selected_feature_i >= self.latent_vector_size:
                self.selected_feature_i = 0


        # WS => add to/remove from selected feature
        move_by = 1.0
        if key_code == "w" or key_code == "s":
            direction = -1.0 # down
            if key_code == "w":
                direction = +1.0  # up

            self.p0[self.selected_feature_i] = self.p0[self.selected_feature_i] + direction * move_by

        #print("feature i=", self.selected_feature_i, ", val=", self.p0[self.selected_feature_i])
        self.p = self.p0

        latents = np.asarray([self.p])
        return latent_to_image_localServerSwitch(latents)

    def start_renderer_key_interact(self):
        self.selected_feature_i = int(self.latent_vector_size / 2.0)
        # self.selected_feature_i = 10 # hmm is there an ordering?
        self.previous = self.p0
        self.SHIFT = False
        self.ALT = False

        self.saved = [None] * 10

        #\self.renderer.show_frames_game(self.get_interpolated_image_key_input)
        self.renderer.show_intro(self.get_interpolated_image_key_input)


interaction_handler = Interaction_Handler()
interaction_handler.latent_vector_size = get_vec_size_localServerSwitch()

version = "v0b"
version = "v2"

steps_speed = 120

if version == "v0":
    interaction_handler.start_renderer_no_interaction()

elif version == "v0b":
    interaction_handler.shuffle_random_points(steps=steps_speed)
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

    interaction_handler.shuffle_random_points(steps=steps_speed)
    interaction_handler.start_renderer_interpolation_interact()

elif version == "v2":
    interaction_handler.shuffle_random_points(steps=1)
    interaction_handler.start_renderer_key_interact()
