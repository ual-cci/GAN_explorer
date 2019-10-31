import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread

import renderer
class Interaction_Handler(object):
    """
    Do all the interaction tricks here.
    """

    def __init__(self, getter):
        self.renderer = renderer.Renderer()
        self.renderer.show_fps = True
        self.getter = getter

        self.latent_vector_size = 512

    # v0 - pure random
    def get_random_image(self, counter):
        how_many = 1
        latents = np.random.randn(how_many, self.latent_vector_size)
        return self.getter.latent_to_image_localServerSwitch(latents)

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
        return self.getter.latent_to_image_localServerSwitch(latents)

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
        return self.getter.latent_to_image_localServerSwitch(latents)

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

        # Render between all saved you have!
        # start with position self.saved[0] and go till self.saved[9]
        # use self.steps
        # save everything we navigate through to /render_interpolation folder


        # Start recording / Stop recording
        # save every new image we get into /renders folder
        # (not saving the reduntant images when we don't move) ... (this will work nicely with "space")

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
        if key_code == "w" or key_code == "s":
            direction = -1.0 # down
            if key_code == "w":
                direction = +1.0  # up

            self.p0[self.selected_feature_i] = self.p0[self.selected_feature_i] + direction * self.move_by

        # +/- change the speed of movement
        if key_code == "+" or key_code == "-":
            mult = 0.5
            if key_code == "+":
                mult = 2.0

            self.move_by = self.move_by * mult
        if key_code == "*": # reset speed of movement
            self.move_by = 1.0


        #print("feature i=", self.selected_feature_i, ", val=", self.p0[self.selected_feature_i])
        self.p = self.p0

        latents = np.asarray([self.p])
        return self.getter.latent_to_image_localServerSwitch(latents)


    def start_renderer_key_interact(self):
        self.selected_feature_i = int(self.latent_vector_size / 2.0)
        # self.selected_feature_i = 10 # hmm is there an ordering?
        self.previous = self.p0
        self.move_by = 1.0
        self.SHIFT = False
        self.ALT = False

        self.saved = [None] * 10

        #\self.renderer.show_frames_game(self.get_interpolated_image_key_input)
        self.renderer.show_intro(self.get_interpolated_image_key_input, self.get_random_image)

