import pickle, pickletools
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import PIL.Image

import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pickle

def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False, shrink=1):
    # Taken from dnnlib/tflib/tfutil.py
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        ksize = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return tf.saturate_cast(images, tf.uint8)

class StyleGAN2_Handler(object):
    """
    Handles a trained StyleGAN2 model
    """

    def __init__(self, settings, args, truncation_psi = 0.9):
        # Initialization, should create the model, load it and also run one inference (to build the graph)
        self.settings = settings
        print("Init handler with path =", args.model_path)
        self.truncation_psi = truncation_psi

        # Load and create a model
        self._create_model(args.model_path)

        self.latent_vector_size = self._Gs.input_shapes[0][1:][0]

        # Infer once to build up the graph (takes like 2 sec extra time is the first one)
        print("Testing with an example to build the graph")
        self.set_noise()
        self.noise_changing = False

        self._example_input = self.example_input(verbose=False)
        self._example_output = self.infer(self._example_input, verbose=False)



    def _create_model(self, model_path):
        tf.InteractiveSession()

        stream = open(model_path, 'rb')

        tflib.init_tf()
        with stream:
            self._G, self._D, self._Gs = pickle.load(stream, encoding='latin1')

    def report(self):

        print("[StyleGAN2_Handler Status report]")
        print("\t- latent_vector_size:",self.latent_vector_size)
        print("\t- typical input shape is:",self._example_input.shape)
        print("\t- typical output shape is:",self._example_output.shape)

    def example_input(self, how_many=1, seed=None, verbose=True):
        if seed:
            if verbose:
                print("Generating random input (from seed=",seed,")")
            latents = np.random.RandomState(seed).randn(how_many, self.latent_vector_size)
        else:
            if verbose:
                print("Generating random input")
            latents = np.random.randn(how_many, self.latent_vector_size)

        # PS: StyleGan2 code used:
        #         rnd = np.random.RandomState(seed)
        #         z = input_latents
        #         print("z my=", z.shape)
        #         z = rnd.randn(1, *self._Gs.input_shape[1:])  # [minibatch, component]

        if verbose:
            print("example input is ...", latents.shape)
        example_input = latents
        return example_input

    def set_noise(self, seed=None):
        noise_vars = [var for name, var in self._Gs.components.synthesis.vars.items() if name.startswith('noise')]
        rnd = np.random.RandomState(seed)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]

    def infer(self, input_latents, verbose=True):

        z = input_latents
        if self.noise_changing:
            self.set_noise()
        #images = self._Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        images = self._Gs.run(z, None, truncation_psi = self.truncation_psi, randomize_noise = False, output_transform = dict(func=convert_images_to_uint8, nchw_to_nhwc=True))  # [minibatch, height, width, channel]

        if verbose:
            print("Generated",images.shape)

        return images

    def save_image(self, image, name="foo.jpg"):
        im = PIL.Image.fromarray(image)
        im.save(name)

    def toggleStylegan2Noise(self):
        self.noise_changing = not self.noise_changing

    def reconnect(self, target_tensor, percent_change = 30):
        print("!!! To be implemented for StyleGAN2 handler !!!")


"""
# Example of usage:

settings = {}
import mock
args = mock.Mock()
settings = mock.Mock()

args.model_path = "../stylegan2/stylegan2-ffhq-config-f.pkl"
args.truncation_psi = 0.5

style2_handler = StyleGAN2_Handler(settings, args)

style2_handler.report()

example_input = style2_handler.example_input()
example_output = style2_handler.infer(example_input)

print("example_output:", example_output.shape)


from timeit import default_timer as timer

# Basic measurements

repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()

    example_input = style2_handler.example_input(verbose=False)
    example_output = style2_handler.infer(example_input, verbose=False)

    t_infer = timer() - t_infer
    if repeat_i > 0:
        times.append(t_infer)
    #print("Prediction (of 1 sample) took", t_infer, "sec.")

times = np.asarray(times)
print("Statistics:")
print("prediction time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")

# Batch measurements

how_many = 4 # too big? gpu mem explodes
repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()

    example_inputs = style2_handler.example_input(how_many = how_many, verbose=False)
    example_outputs = style2_handler.infer(example_inputs, verbose=False)

    t_infer = timer() - t_infer

    if repeat_i > 0:
        times.append(t_infer)
    #print("Prediction (of",how_many,"samples) took", t_infer, "sec.")

times = np.asarray(times)
print("Statistics:")
print("prediction of whole",how_many," took time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")
print("prediction as divided for one - avg +- std =", np.mean(times/how_many), "+-", np.std(times/how_many), "sec.")
"""