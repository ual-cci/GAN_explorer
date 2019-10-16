import requests
from timeit import default_timer as timer
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

# = HANDSHAKE =================================================
PORT = "5000"
Handshake_REST_API_URL = "http://localhost:"+PORT+"/handshake"

payload = {"client": "client", "backup_name":"Bob"}
r = requests.post(Handshake_REST_API_URL, files=payload).json()
print("Handshake request data", r)

# = SEND BATCH OF IMAGES =====================================

PORT = "5000"
Images_REST_API_URL = "http://localhost:"+PORT+"/get_image"
"""
while 1:
    payload = {}
    how_many = 1
    latent_vector_size = 512
    latents = np.random.randn(how_many, latent_vector_size)
    # image = open(IMAGE_PATH, "rb").read()
    payload["latents"] = latents.tolist()

    # submit the request
    start = timer()
    r = requests.post(Images_REST_API_URL, json=payload).json()

    total_time = timer() - start
    print("Time total", total_time)
    # print("request data", r)
    # for i,item in enumerate(r['results']):
    #    print(r['uids'][i]," = len results", len(item), item)

    print("1 time_decode", r["time_decode"])
    print("2 time_infer", r["time_infer"])
    print("3 time_encode", r["time_encode"])
    print("4 jasonify (from logs)", "JSONify took 0.2742667689999507 sec.")
    print("time_server_total", r["time_server_total"])

    time_communication = total_time - r["time_server_total"]
    print("time_communication", time_communication)

    images_returned = np.array(r['images_response'])
    print("images_returned:", images_returned.shape)
    for image in images_returned:
        from utils.image_utils import toimage

        image = toimage(image)

        imgplot = plt.imshow(image)
        plt.show()
"""

Images_REST_API_URL_2 = "http://localhost:"+PORT+"/get_image_2"

from PIL import Image
import requests
from io import BytesIO
while 1:
    payload = {}
    how_many = 1
    latent_vector_size = 512
    latents = np.random.randn(how_many, latent_vector_size)
    # image = open(IMAGE_PATH, "rb").read()
    payload["latents"] = latents.tolist()

    # submit the request
    start = timer()

    r = requests.post(Images_REST_API_URL_2, json=payload)

    # open as compressed (slower than jpeg, but faster than png)
    """
    buf = BytesIO(r.content)
    npzfile = np.load(buf)

    images = npzfile['arr_0']  # default names are given unless you use keywords to name your arrays
    img = images[0]
    """

    # actual file
    img = Image.open(BytesIO(r.content))

    total_time = timer() - start
    print("Time total", total_time)

    imgplot = plt.imshow(img)
    plt.show()