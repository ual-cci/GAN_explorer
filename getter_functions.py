import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

import progressive_gan_handler
from settings import Settings


class Getter(object):
    """
    Handles function to get images from handler (running either locally or on server).
    """

    def __init__(self, args, USE_SERVER_INSTEAD = False):

        self.USE_SERVER_INSTEAD = USE_SERVER_INSTEAD

        if self.USE_SERVER_INSTEAD:
            # = HANDSHAKE =================================================
            PORT = "8000"
            self.Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

            payload = {"client": "client", "backup_name":"Bob"}
            r = requests.post(self.Handshake_REST_API_URL, files=payload).json()
            print("Handshake request data", r)

            # = SEND BATCH OF IMAGES =====================================
            self.Images_REST_API_URL = "http://localhost:"+PORT+"/get_image"


        serverside_handler = None

        settings = Settings()
        self.serverside_handler = None

        if not self.USE_SERVER_INSTEAD:
            self.serverside_handler = progressive_gan_handler.ProgressiveGAN_Handler(settings, args)

    def get_image_directly(self, latents):
        # start = timer()

        images = self.serverside_handler.infer(latents, verbose=False)

        # total_time = timer() - start
        # print("Time total", total_time)

        return images[0]

    def get_image_from_server(self, latents):
        # image = open(IMAGE_PATH, "rb").read()
        payload = {}
        payload["latents"] = latents.tolist()

        # submit the request
        start = timer()

        r = requests.post(self.Images_REST_API_URL, json=payload)

        # open as compressed (slower than jpeg, but faster than png)
        """
        buf = BytesIO(r.content)q
        npzfile = np.load(buf)

        images = npzfile['arr_0']  # default names are given unless you use keywords to name your arrays
        img = images[0]
        """

        # actual file
        img = Image.open(BytesIO(r.content))

        total_time = timer() - start
        print("Time total", total_time)

        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        print("open_cv_image", open_cv_image.shape)

        return open_cv_image

    def latent_to_image_localServerSwitch(self, latents):
        """
        This function handles which version we are doing (on server vs. locally)
        """
        if self.USE_SERVER_INSTEAD:
            img = self.get_image_from_server(latents)
        else:
            img = self.get_image_directly(latents)


        # Model produces BGR
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_vec_size_localServerSwitch(self):
        """
        This function handles which version we are doing (on server vs. locally)
        """
        vec_size = -1
        if self.USE_SERVER_INSTEAD:
            print("TO IMPLEMENT")
            assert False
        else:
            vec_size = self.serverside_handler.latent_vector_size
        return vec_size

