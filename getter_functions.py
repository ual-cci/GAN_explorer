import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

USE_SERVER_INSTEAD = False
#USE_SERVER_INSTEAD = True # < slower

if USE_SERVER_INSTEAD:
    # = HANDSHAKE =================================================
    PORT = "8000"
    Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

    payload = {"client": "client", "backup_name":"Bob"}
    r = requests.post(Handshake_REST_API_URL, files=payload).json()
    print("Handshake request data", r)

# = SEND BATCH OF IMAGES =====================================

PORT = "8000"
Images_REST_API_URL = "http://localhost:"+PORT+"/get_image"


serverside_handler = None
import progressive_gan_handler
import settings

settings = settings.Settings()

import mock

args = mock.Mock()
args.model_path = 'aerials512vectors1024px_snapshot-010200.pkl'
serverside_handler = None

if not USE_SERVER_INSTEAD:
    serverside_handler = progressive_gan_handler.ProgressiveGAN_Handler(settings, args)

def get_image_directly(latents):
    global serverside_handler
    #start = timer()

    images = serverside_handler.infer(latents)

    #total_time = timer() - start
    #print("Time total", total_time)

    return images[0]


def get_image_from_server(latents):
    # image = open(IMAGE_PATH, "rb").read()
    payload = {}
    payload["latents"] = latents.tolist()

    # submit the request
    start = timer()

    r = requests.post(Images_REST_API_URL, json=payload)

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

def latent_to_image_localServerSwitch(latents):
    """
    This function handles which version we are doing (on server vs. locally)
    """
    if USE_SERVER_INSTEAD:
        img = get_image_from_server(latents)
    else:
        img = get_image_directly(latents)
    return img

"""
while 1:
    img = get_image()
    imgplot = plt.imshow(img)
    plt.show()
"""