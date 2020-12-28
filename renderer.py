import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread

import time
import collections
import skimage.transform


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

    def __init__(self, show_fps = True, initial_resolution = 1024):
        self.show_fps = show_fps
        self.counter = 0
        self.initial_resolution = initial_resolution

        #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        #cv2.resizeWindow('frame', initial_resolution, initial_resolution)

        return None

    def show_frames(self, get_image_function):
        fps = FPS()
        self.counter = 0

        while (True):
            self.counter += 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('v'):
                self.show_fps = not self.show_fps

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

            cv2.imshow('Interactive Machine Learning - GAN', frame)


    def show_frames_game(self, get_image_function):
        # This function is made with WASD and SPACE control scheme in mind
        fps = FPS()
        self.counter = 0
        message = ""

        while (True):
            self.counter += 1

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('v'):
                self.show_fps = not self.show_fps

            if key is not -1:
                print(key)

            key_code = ""
            nums = [str(i) for i in list(range(0,10))]
            allowed_keys = ["w","s","a","d", # movement
                            "n", " ",
                            "f","g","t","h","u","j", # nn hacks and restore (j)
                            "y", # nn hacks - simulation of a normal random weights replacement
                            "l", "k", # save load latents
                            "m", # reorder latents
                            "p", # plots
                            "r", # random jump
                            "e",
                            "o", # debug key to run custom commands
                            "+", "-", "*",
                            "=", # interpolate
                            "]", 'x', "z"] + nums
            allowed_keys_ord = [ord(k) for k in allowed_keys]
            if key in allowed_keys_ord:
                key_code = chr(key)
            else:
                # DEBUG PART for key press detections:
                if key != -1:
                    print("not allowed key detected:",key)
                    try:
                        print("this might be",chr(key),key)
                    except Exception as e:
                        print("err trying to call chr(key)", e)

            if key == 190: # pressed F1 for help!
                self.print_help()

            # get image
            frame = None

            time_start = timer()
            frame, new_message = get_image_function(self.counter, key_code, key)
            if len(new_message) > 0:
                message = new_message

            time_end = timer()
            #print("timer:", (time_end-time_start))
            #fps_val = 1.0 / (time_end-time_start)
            fps_val = fps()
            # includes time for the open cv to render and the text
            # so without showing it (imshow), it would be 24-26fps, with it it's cca 20-21fps
            #print(fps_val)

            if self.show_fps:
                if len(message) > 0:
                    text = message
                    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    textX = int((frame.shape[1] - textsize[0]) / 2)
                    textY = int((frame.shape[0] + textsize[1]) / 2)

                    frame = cv2.putText(frame, text, (self.initial_resolution - textsize[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    #frame = cv2.putText(frame, text, (textX, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                frame = cv2.putText(frame, "FPS " + '{:.2f}'.format(fps_val), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Interactive Machine Learning - GAN', frame)

    def make_a_grid(self, get_image_function, x_times = 4, y_times = 4, resolution_of_one = 256):

        z = 3
        frame = np.ones((x_times*resolution_of_one, y_times*resolution_of_one, z), np.float)
        #print("frame ->", frame.shape)

        counter = 0

        for i in range(x_times):
            for j in range(y_times):

                image = get_image_function(0)
                #return image
                image_resized = skimage.transform.resize(image, (resolution_of_one, resolution_of_one))
                #print("image_resized ->", image_resized.shape)

                # numpy indexing: rows, columns
                jump_x = i*resolution_of_one
                jump_y = j*resolution_of_one
                frame[0+jump_x:resolution_of_one+jump_x, 0+jump_y:resolution_of_one+jump_y] = image_resized[:,:]
                # awful frame[0+jump_x:resolution_of_one+jump_x, 0+jump_y:resolution_of_one+jump_y, counter%3] = image_resized[:,:, 0]

                counter += 1

        #frame = frame * 0.8
        return frame

    def show_intro(self, get_image_function, get_grid_image_function):
        resolution = 1024
        end = False

        # Generate background as a grid of samples
        times = 6
        frame = self.make_a_grid(get_grid_image_function, times, times, int(resolution/times))

        # Layer over the text
        self.texts = ["<< GAN interaction Game >>", "",
                 "Controls:",
                 " - ws: move forwards/backwards",
                 " - ad: change direction",
                 " - space: small perturbation of the space",
                 " - r: randomly place elsewhere",
                 " - SHIFT: toggles save or load",
                 " - 0-9: save to/load from a slot number 0-9",
                 " - v: FPS on/off",
                 " - fg (t) h: NN hacks",
                 "", "", "", "", "", "", "", "", "[[ Press space to continue ... ]]"]


        while (True):
            key = cv2.waitKey(1)
            if key == ord('q'):
                end = True
                break
            if key == ord(' '):
                end = False
                break
            if key == ord('v'):
                self.show_fps = not self.show_fps

            frame_dynamic = frame.copy()

            #times = 3
            #frame_dynamic = self.make_a_grid(get_grid_image_function, times, times, int(resolution / times))

            if self.show_fps:
                left = 100
                top = 140
                title = True
                for text in self.texts:
                    thickness = 2
                    if title:
                        thickness = 3
                    frame_dynamic = cv2.putText(frame_dynamic, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), thickness)
                    title = False
                    top += 40

            cv2.imshow('Interactive Machine Learning - GAN', frame_dynamic)

        if not end:
            # Continue!
            self.show_frames_game(get_image_function)

    def print_help(self):
        print("\n".join(self.texts))