import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread
import os, os.path
import glob

import renderer
import reconnector
import plotter

OSC_HANDLER = None
SIGNAL_interactive_i = 0.0
SIGNAL_reset_toggle = 0
SIGNAL_latents = []


class Interaction_Handler(object):
    """
    Do all the interaction tricks here.
    """

    def __init__(self, getter, initial_resolution=1024):
        self.renderer = renderer.Renderer(initial_resolution=initial_resolution)
        self.renderer.show_fps = True
        self.getter = getter
        self.game_is_in_interpolating_mode = False
        self.saved_first_index_selected = -1
        self.saved_second_index_selected = -1
        self.counter_start = 0

        self.latent_vector_size = 512
        self.saved_already = 0

        self.keep_p1 = False # special case for v0b

        self.toggle_save_frames_in_loop = False

        # network hacking
        self.multiplier_value = 1.0
        self.target_tensor = 0
        ##self.target_tensors = ["16x16/Conv0_up/weight", "32x32/Conv0_up/weight", "64x64/Conv0_up/weight", "128x128/Conv0_up/weight", "256x256/Conv0_up/weight"]
        self.target_tensors = ["16x16/Conv0/weight", "32x32/Conv0/weight", "64x64/Conv0/weight", "128x128/Conv0/weight", "256x256/Conv0/weight"] # << Pre-trained PGGAN has these

        self.convolutional_layer_reconnection_strength = 0.3

        # plotting:
        self.plotter = plotter.Plotter(self.renderer, getter)

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

        if self.keep_p1:
            self.tmp_save_p1 = self.p1

        latents = np.random.randn(how_many, self.latent_vector_size)

        self.p0 = latents[0]
        if self.keep_p1:
            self.p0 = self.tmp_save_p1

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

        global OSC_HANDLER
        if OSC_HANDLER is None:
            OSC_address = '0.0.0.0'
            OSC_port = 8000
            OSC_bind = b'/send_gan_i'
            global SIGNAL_interactive_i
            global SIGNAL_reset_toggle
            global SIGNAL_latents

            # OSC - Interactive listener
            def callback(*values):
                global SIGNAL_interactive_i
                global SIGNAL_reset_toggle
                global SIGNAL_latents
                #print("OSC got values: {}".format(values))

                percentage = values[0]
                reset_toggle = values[1]
                signal_latent = values[2:]

                SIGNAL_interactive_i = 0 #float(percentage) / 1000.0  # 1000 = 100% = 1.0
                SIGNAL_reset_toggle = 1

                #print("signal_latent len=", len(signal_latent))
                signal_latent = np.asarray(signal_latent)
                SIGNAL_latents.append( signal_latent )

                THRESHOLD_NUMBER_of_last_latents = 30
                if len(SIGNAL_latents) > THRESHOLD_NUMBER_of_last_latents:
                    SIGNAL_latents = SIGNAL_latents[-THRESHOLD_NUMBER_of_last_latents:]

            print("Also starting a OSC listener at ", OSC_address, OSC_port, OSC_bind,
                  "to listen for interactive signal (0-1000).")
            from oscpy.server import OSCThreadServer
            osc = OSCThreadServer()
            sock = osc.listen(address=OSC_address, port=OSC_port, default=True)
            osc.bind(OSC_bind, callback)

            OSC_HANDLER = osc # No longer none


        # ignore counter
        global SIGNAL_interactive_i
        global SIGNAL_reset_toggle
        global SIGNAL_latents

        if SIGNAL_reset_toggle == 1:
            self.p0 = self.p
            if len(SIGNAL_latents) > 0:
                latent = SIGNAL_latents[0]
                SIGNAL_latents = SIGNAL_latents[1:]
                self.p1 = np.asarray(latent)

        alpha = float(SIGNAL_interactive_i) / 30.0
        SIGNAL_interactive_i += 1 # hmmmm easy interpolation test
        self.p = self.p0 + (alpha) * (self.p1 - self.p0)
        latents = np.asarray([self.p])

        return self.getter.latent_to_image_localServerSwitch(latents)

    def start_renderer_interpolation_interact(self):
        self.renderer.show_frames(self.get_interpolated_image_OSC_input)

    def select_saved_latents(self):
        suceeded = False

        last_index = self.saved_second_index_selected

        total_saved_latents = 0
        valid_indices = []
        for idx in range(len(self.saved)):
            if self.saved[idx] is not None:
                total_saved_latents += 1
                valid_indices.append(idx)

        print("We have", total_saved_latents, "total saved latents to interpolate between. valid_indices=",valid_indices)

        if total_saved_latents >= 2:
            suceeded = True

            if last_index is -1: # aka this is for the first time
                self.saved_first_index_selected = valid_indices[0]
                self.saved_second_index_selected = valid_indices[1]
            else:
                position_of_the_last_one = valid_indices.index(last_index)
                self.saved_first_index_selected = valid_indices[position_of_the_last_one]

                position_of_the_second_one = position_of_the_last_one + 1
                if position_of_the_second_one >= len(valid_indices):
                    position_of_the_second_one = 0

                self.saved_second_index_selected = valid_indices[position_of_the_second_one]

            print("set indices as self.saved_first_index_selected=",self.saved_first_index_selected,", self.saved_second_index_selected=",self.saved_second_index_selected)

        if (not last_index == -1) and valid_indices[1] == self.saved_second_index_selected:
            print("LAST FRAME")
            last_index = -1
            self.toggle_save_frames_in_loop = False

        if suceeded:
            self.set_saved_values_as_latents_to_interpolate(self.saved_first_index_selected, self.saved_second_index_selected)
        else:
            # if we don't have anything to intepolate between ... don't do it
            self.game_is_in_interpolating_mode = False

    def set_saved_values_as_latents_to_interpolate(self, first_i, second_i):
        self.p0 = self.saved[first_i]
        self.p1 = self.saved[second_i]

    # v2 - controlled using wsad
    def get_interpolated_image_key_input(self, counter, key_code, key_ord):
        # ignore counter
        # look at the key command
        #if key_ord is not -1:
        #    print("key pressed (code, ord)", key_code, key_ord)
        message = ""
        save_frame_to_file = False

        # Save & Load - shift or z
        if key_ord is 225 or key_ord is 226 or key_code == "z":
            self.SHIFT = not self.SHIFT
            print("Saving ON?:", self.SHIFT)
        if key_ord is 233 or key_ord is 234:
            self.ALT = not self.ALT

        nums = [str(i) for i in list(range(0,9))]

        if self.SHIFT and key_code in nums:
            # SAVE on position
            save_to_i = int(key_code)
            print("saving to ", save_to_i)
            message = "Saved to "+str(save_to_i)
            self.saved[save_to_i] = self.p0
        if not self.SHIFT and key_code in nums:
            # LOAD from position
            load_from_i = int(key_code)
            print("loading from ", load_from_i)
            message = "Loading from " + str(load_from_i)
            if self.saved[load_from_i] is not None:
                self.p0 = self.saved[load_from_i]

        # Render between all saved you have!
        # start with position self.saved[0] and go till self.saved[9]
        # use self.steps
        # save everything we navigate through to /render_interpolation folder
        if key_code == "=":
            # Start interpolation
            self.game_is_in_interpolating_mode = not self.game_is_in_interpolating_mode

            if self.game_is_in_interpolating_mode:
                self.counter_start = counter  # so that we always start with 0
                self.saved_first_index_selected = -1
                self.saved_second_index_selected = -1 # restart from the first one again
            else:
                self.p0 = self.p # to start where we now ended at

        if key_code == "]":
            self.toggle_save_frames_in_loop = not self.toggle_save_frames_in_loop
            print("When interpolating with =, save frames =",self.toggle_save_frames_in_loop)

        if self.game_is_in_interpolating_mode:
            if self.toggle_save_frames_in_loop:
                save_frame_to_file = True

            message = "Interpolation"
            self.step = (counter-self.counter_start) % self.steps
            #print(counter, counter-self.counter_start, self.step, "from", self.steps)

            if self.step == 0:
                self.select_saved_latents()
                #self.shuffle_random_points(self.steps)
            self.p = self.calculate_p(step=self.step)



        # Start recording / Stop recording
        # save every new image we get into /renders folder
        # (not saving the reduntant images when we don't move) ... (this will work nicely with "space")

        if key_code == "x":
            save_frame_to_file = True

        if key_code == "n":
            print("Noise relevant to StyleGan2 toggled")
            self.getter.toggleStylegan2Noise()


        # Random jump
        if key_code == "r":
            message = "Random"
            self.previous = self.p0
            self.shuffle_random_points(self.steps)

        # One undo
        if key_code == "e":
            message = "Previous"
            tmp = self.p0
            self.p0 = self.previous
            self.previous = tmp

        # small jump
        if key_code == " ":
            message = "Jump"
            save_p0 = self.p0
            self.shuffle_random_points(self.steps)

            alpha = 0.02
            self.p0 = (1.0 - alpha) * save_p0 + alpha*self.p0

        # AD => selecting a feature (0 to self.latent_vector_size and then loop)
        if key_code == "a" or key_code == "d":
            message = "a"

            direction = -1 # left
            if key_code == "d":
                message = "d"

                direction = +1  # right
            self.selected_feature_i = self.selected_feature_i + direction


            if self.selected_feature_i < 0:
                self.selected_feature_i = self.latent_vector_size - 1
            if self.selected_feature_i >= self.latent_vector_size:
                self.selected_feature_i = 0



            # combine A-D with W-S ? optional
            """
            direction = -1.0 # down - d
            if key_code == "a":
                direction = +1.0  # up - a
            self.p0[self.selected_feature_i] = self.p0[self.selected_feature_i] + direction * self.move_by
            """


        # WS => add to/remove from selected feature
        if key_code == "w" or key_code == "s":
            message = "s"

            direction = -1.0 # down
            if key_code == "w":
                message = "w"

                direction = +1.0  # up

            self.p0[self.selected_feature_i] = self.p0[self.selected_feature_i] + direction * self.move_by

        # f / g Allow editing of the GAN network weights on the fly!
        # t swaps which tensor this influences

        if key_code == "f":
            target_tensor = self.target_tensors[self.target_tensor]
            #self.multiplier_value += 0.1
            self.multiplier_value *= 1.5
            print("self.multiplier_value", self.multiplier_value)
            function = self.getter.serverside_handler.times_a
            self.getter.serverside_handler.change_net(target_tensor, function, self.multiplier_value)

        if key_code == "g":
            target_tensor = self.target_tensors[self.target_tensor]
            self.multiplier_value /= 1.5
            print("self.multiplier_value", self.multiplier_value)
            function = self.getter.serverside_handler.times_a
            self.getter.serverside_handler.change_net(target_tensor, function, self.multiplier_value)

        if key_code == "t":
            self.target_tensor += 1
            if self.target_tensor >= len(self.target_tensors):
                self.target_tensor = 0
            print("selected target_tensor", self.target_tensors[self.target_tensor])

        if key_code == "h":
            print("RECONNECTOR")
            target_tensor = self.target_tensors[self.target_tensor]

            #target_tensor = "16x16/Conv0_up/weight"  # "128x128/Conv0_up/weight"
            percent_change = 100.0*self.convolutional_layer_reconnection_strength #(0.3)*100=30
            #percent_change = 15
            self.getter.serverside_handler.reconnect(target_tensor, percent_change)
            print("Reconnected", percent_change,"% of conv kernels in", target_tensor)

        if key_code == "j":
            print("RECONNECTOR reset")
            self.getter.serverside_handler.restore()

        if key_code == "u":
            print("HAXED NET SAVER")
            self.getter.serverside_handler.savenet()

        # +/- change the speed of movement
        if key_code == "+" or key_code == "-":
            mult = 0.5
            if key_code == "+":
                mult = 2.0

            self.move_by = self.move_by * mult
        if key_code == "*": # reset speed of movement
            self.move_by = 1.0

        if key_code == "m":
            print("Randomizing the order!")
            self.randomize_saved_order()

        # Save existing latents into a folder:
        if key_code == "k":  # save latents
            print("Saving latents!")
            #print("Saving:", self.saved)
            path = "latents/"
            if not os.path.exists(path):
                os.mkdir(path)
            path = "latents/save/"
            if not os.path.exists(path):
                os.mkdir(path)

            # get name of the last file in the folder and append
            max_file_int = 0
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    file_int = int( file.replace(".txt", "") )
                    max_file_int = max(max_file_int, file_int)
            max_file_int+=1

            for latent in self.saved:
                if latent is not None:
                    target = path+str(max_file_int).zfill(5)
                    print("Saving to", target)

                    np.savetxt(target+".txt", latent)

                    img_tmp = self.getter.latent_to_image_localServerSwitch(np.asarray([latent]))
                    cv2.imwrite(target+".png", img_tmp)
                    del img_tmp

                    max_file_int += 1

        # Load latents from a folder:
        if key_code == "l": # load latents
            self.saved = []
            path = "latents/"
            if not os.path.exists(path):
                os.mkdir(path)
            path = "latents/load/"
            if not os.path.exists(path):
                os.mkdir(path)

            for file in os.listdir(path):
                if file.endswith(".txt"):
                    latent = np.loadtxt(os.path.join(path, file))
                    self.saved.append(latent)

            print("Loaded in total:", len(self.saved))
            while len(self.saved) < 10:
                self.saved.append(None)

        #print("feature i=", self.selected_feature_i, ", val=", self.p0[self.selected_feature_i])
        if not self.game_is_in_interpolating_mode:
            self.p = self.p0

        # Plotting:
        if key_code == "p": # plot
            self.plotter.plot(self.p)


        latents = np.asarray([self.p])

        image = self.getter.latent_to_image_localServerSwitch(latents)


        # Simple save in HQ (post-render)
        if save_frame_to_file:
            # Single file save:
            #"""
            message = "Saved file"
            folder = "renders/"
            if not os.path.exists(folder):
                os.mkdir(folder)

            filename = folder+"saved_" + str(self.saved_already).zfill(4) + ".png"
            self.saved_already += 1
            print("Saving in good quality as ", filename)

            cv2.imwrite(filename, image)
            #"""

            # Single plot save!
            """
            print("-----------------", self.saved_already)
            self.plotter.plot(self.p, counter_override = self.saved_already)
            self.saved_already += 1
            print("-----------------//end",)
            """

        return image, message


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

    def randomize_saved_order(self):
        # Randomize the order of current latents (ones in self.saved) (they can be loaded by "l" and re-saved again by "k")
        np.random.shuffle(self.saved)
