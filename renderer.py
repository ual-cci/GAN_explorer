import cv2
import numpy as np
from timeit import default_timer as timer
import datetime, time
from threading import Thread

import time
import collections
import skimage.transform

FRAME_NAME = 'Interactive Machine Learning - GAN'

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

    def __init__(self, show_fps = True, initial_resolution = 1024, fullscreen='None'):
        self.show_fps = show_fps
        self.counter = 0
        self.initial_resolution = initial_resolution
        self.fullscreen = fullscreen
        self.screen_width, self.screen_height = None, None

        self.window_created = False

        return None

    def create_window(self):
        # Create named window, set all the needed cv2 flags
        if self.fullscreen=='resize':
            cv2.namedWindow(FRAME_NAME, cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
            cv2.resizeWindow(FRAME_NAME, initial_resolution, initial_resolution)

        if self.fullscreen=='full':
            cv2.namedWindow(FRAME_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(FRAME_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # we need to know the screen dimensions for this ...
            self.screen_width, self.screen_height = 1920, 1080


    def trigger_render(self, frame):
        # Render function - either just diplays the image, or also adds borders
        if not self.window_created:
            self.create_window()
            self.window_created = True

        if self.fullscreen=='full':
            # we need to add black borders:

            # border: https://stackoverflow.com/questions/36255654/how-to-add-border-around-an-image-in-opencv-python
            # display number: https://stackoverflow.com/a/53005272 ~ perhaps cv2.moveWindow(window_name, *first_display_size)
            frame_w, frame_h, _ = frame.shape # 1020, 1020, 3

            black = [0,0,0]
            top_bottom = int( (self.screen_height - frame_h) / 2 )
            left_right = int( (self.screen_width - frame_w) / 2 )
            frame = cv2.copyMakeBorder(frame,top_bottom,top_bottom,left_right,left_right,cv2.BORDER_CONSTANT,value=black)

            cv2.imshow(FRAME_NAME, frame)
        else:
            # normally call imshow
            cv2.imshow(FRAME_NAME, frame)

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

            self.trigger_render(frame)

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
                            "`", # autonomous mode
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

            self.trigger_render(frame)

    def return_frame_client_mode(self, get_image_function, key_code):
        # This function is made with WASD and SPACE control scheme in mind
        fps = FPS()
        self.counter = 0
        message = ""


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
        if key_code not in allowed_keys:
                print("not allowed key detected:",key_code)

        # get image
        frame = None

        time_start = timer()
        frame, new_message = get_image_function(self.counter, key_code, ord(key_code))
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

        return frame

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

            if True: # always show the initial message
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

            self.trigger_render(frame_dynamic)

        if not end:
            # Continue!
            self.show_frames_game(get_image_function)

    def print_help(self):
        print("\n".join(self.texts))