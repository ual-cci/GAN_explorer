from threading import Thread
import time

from PIL import Image
import flask
import os
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool
import numpy as np
import socket
import cv2
import progressive_gan_handler
import settings
import io
import PIL.Image

# Thanks to the tutorial at: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

# Production vs development - not entirely sure what the differences are, except for debug outputs
PRODUCTION = True # waitress - production-quality pure-Python WSGI server with very acceptable performance
PRODUCTION = False # Flask

SERVER_VERBOSE = 2  # 2 = all messages, 1 = only important ones, 0 = none!

if PRODUCTION:
    from waitress import serve

app = flask.Flask(__name__)
serverside_handler = None
pool = ThreadPool()

PORT = 8000

class Server(object):
    """
    Server
    """

    def __init__(self, args):
        print("Server ... starting server and loading model ... please wait until its started ...")

        self.settings = settings.Settings()
        self.load_serverside_handler(args)

        frequency_sec = 10.0
        if SERVER_VERBOSE > 0:
            t = Thread(target=self.mem_monitor_deamon, args=([frequency_sec]))
            t.daemon = True
            t.start()

        # hack to distinguish server by hostnames
        hostname = socket.gethostname()  # gpu048.etcetcetc.edu
        print("server hostname is", hostname)

        if PRODUCTION:
            serve(app, host='127.0.0.1', port=PORT)
        else:
            app.run(threaded=False, port=PORT) # < with forbiding threaded we still have the same default graph

    def mem_monitor_deamon(self, frequency_sec):
        import subprocess
        while (True):
            out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                                   stdout=subprocess.PIPE).communicate()[0].split(b'\n')
            vsz_index = out[0].split().index(b'RSS')
            mem = float(out[1].split()[vsz_index]) / 1024

            print("Memory:", mem)
            time.sleep(frequency_sec)  # check every frequency_sec sec

    def load_serverside_handler(self, args):
        global serverside_handler
        serverside_handler = progressive_gan_handler.ProgressiveGAN_Handler(self.settings, args)
        print('Server GAN handler loaded.')



@app.route("/handshake", methods=["POST"])
def handshake():
    # Handshake

    data = {"success": False}
    start = timer()

    if flask.request.method == "POST":
        if flask.request.files.get("client"):
            client_message = flask.request.files["client"].read()
            print("Handshake, received: ",client_message)

            backup_name = flask.request.files["backup_name"].read()
            # try to figure out what kind of server we are, what is our name, where do we live, what are we like,
            # which gpu we occupy
            # and return it at an identifier to the client ~

            try:
                hostname = socket.gethostname() # gpu048.etcetcetc.edu
                machine_name = hostname.split(".")[0]
                data["server_name"] = machine_name
            except Exception as e:
                data["server_name"] = backup_name

            end = timer()
            data["internal_time"] = end - start
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route("/get_image", methods=["POST"])
def get_image():
    # Evaluate data
    data = {"success": False}
    if flask.request.method == "POST":
        t_decode_start = timer()

        """
        DEFAULT_interactive_i = 0.0
        DEFAULT_model_i = 0
        DEFAULT_song_i = 0
        interactive_i = DEFAULT_interactive_i
        model_i = DEFAULT_model_i
        song_i = DEFAULT_song_i
        """
        DEFAULT_latents = None
        latents = DEFAULT_latents

        if len(flask.request.files) and SERVER_VERBOSE > 1:
            print("Recieved flask.request.files = ",flask.request.files)

        try:
            #latents = flask.request.files["latents"].read()
            data = flask.request.json
            latents = np.array(data['latents'])
            print(latents.shape)


        except Exception as e:
            print("failed to read the sent latents", e)

        print("Server will generate image from the requested latents",latents.shape)

        t_decode_end = timer()

        global serverside_handler
        t_infer_start = timer()
        images = serverside_handler.infer(latents)
        t_infer_end = timer()

        print("time_infer =",t_infer_end-t_infer_start)
        print("time_decode =",t_decode_end-t_decode_start)


        # Save into file - slightly slower
        #serverside_handler.save_image(images[0])

        filetype = 'jpeg'
        #filetype = 'png' # < Slower than np.savez_compressed !
        # Save into buffer
        image = images[0]
        buf = io.BytesIO() # create our buffer
        image = PIL.Image.fromarray(image)
        image.save(buf, filetype)
        buf.name = 'foo.'+filetype
        buf.seek(0) # restart the file

        return send_file(buf, mimetype='image/'+filetype)


    """
    # using compressed save is slower (maybe slightly better q)
    import io
    buf = io.BytesIO()  # create our buffer
    # pass the buffer as you would an open file object
    np.savez_compressed(buf, images)
    buf.seek(0)  # This simulates closing the file and re-opening it.
    #  Otherwise the cursor will already be at the end of the
    #  file when flask tries to read the contents, and it will
    #  think the file is empty.
    return send_file(buf, mimetype='image/jpeg')
    """

@app.route("/debugMethod", methods=["GET"])
def debugMethod():
    # This just does something I want to test...
    data = {"success": False}
    try:
        global serverside_handler
        # Do something here ...

        # indicate that the request was a success
        data["success"] = True
    except Exception as e:
        print("something went wrong!", e)

    as_json = flask.jsonify(data)
    return as_json

from flask import send_file

@app.route('/get_image_file')
def get_image_file():
    # kinda debug
    filename = 'foo.jpg'
    return send_file(filename, mimetype='image/jpeg')


def get_gpus_buses():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
    buses = ""
    for device in gpu_devices:
        desc = device.physical_device_desc # device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:81:00.0
        bus = desc.split(",")[-1].split(" ")[-1][5:] # split to get to the bus information
        bus = bus[0:2] # idk if this covers every aspect of gpu bus
        if len(buses)>0:
            buses += ";"
        buses += str(bus)
    return buses

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Project: Real Time Image Generation.')
    parser.add_argument('-foo', help='foo value', default='666')
    parser.add_argument('-model_path', help='model_path', default='models/aerials512vectors1024px_snapshot-010200.pkl')
    args = parser.parse_args()

    ##args.model_path = "models/aerials512vectors1024px_snapshot-010200.pkl"

    server = Server(args)