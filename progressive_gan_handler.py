import pickle, pickletools
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
import PIL.Image


class ProgressiveGAN_Handler(object):
    """
    Handles a trained Progressive GAN model, loads from
    """

    def __init__(self, model_path):
        # Initialization, should create the model, load it and also run one inference (to build the graph)
        print("Init handler with path =", model_path)

        # Load and create a model
        #self._create_model(model_path)

        self.latent_vector_size = 512 # get it from the model
        #self.latent_vector_size = self._Gs.input_shapes[0][1:]

        self._mode_intermediate_output = False
        self._mode_intermediate_output_layer_name = "" # "Gs/Grow_lod1/add:0"

        # Infer once to build up the graph (takes like 2 sec extra time is the first one)
        print("Testing with an example to build the graph")
        self._example_input = self.example_input(verbose=False)
        self._example_output = self.infer(self._example_input, verbose=False)

    def _create_model(self, model_path):
        tf.InteractiveSession()
        with open(model_path, 'rb') as file:
            print(file)

            proto_op = next(pickletools.genops(file))
            assert proto_op[0].name == 'PROTO'
            proto_ver = proto_op[1]
            print("Pickled with version", proto_ver)

            self._G, self._D, self._Gs = pickle.load(file)

    def report(self):

        print("[ProgressiveGAN_Handler Status report]")
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

        if verbose:
            print("example input is ...", latents.shape)
        example_input = latents
        return example_input

    def infer(self, input_latents, verbose=True):
        labels = np.zeros([input_latents.shape[0]] + [self.latent_vector_size])

        if self._mode_intermediate_output:
            images = self._Gs.custom_tinkered_layer_output_run(self._mode_intermediate_output_layer_name,False,input_latents,labels)
        else:
            images = self._Gs.run(input_latents,labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

        if verbose:
            print("Generated",images.shape)

        return images

    def set_mode(self, intermediate_output_layer=None):
        print("Setting mode 'intermediate_output_layer' to", intermediate_output_layer)
        if intermediate_output_layer is None:
            self._mode_intermediate_output = False
        else:
            self._mode_intermediate_output = True
            self._mode_intermediate_output_layer_name = intermediate_output_layer

# Example of usage:

pro_path = "aerials512vectors1024px_snapshot-010200.pkl"
pro_handler = ProgressiveGAN_Handler(pro_path)

pro_handler.report()

example_input = pro_handler.example_input()
example_output = pro_handler.infer(example_input)

print("example_output:", example_output.shape)

from timeit import default_timer as timer

# Basic measurements

repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()

    example_input = pro_handler.example_input()
    example_output = pro_handler.infer(example_input)

    t_infer = timer() - t_infer
    times.append(t_infer)
    print("Prediction (of 1 sample) took", t_infer, "sec.")

times = np.asarray(times)
print("Statistics:")
print("prediction time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")

# Batch measurements

how_many = 60 # thinking of showing about nearby 60 per sec - 60 fps?
repeats = 15
times = []
for repeat_i in range(repeats):
    t_infer = timer()

    example_inputs = pro_handler.example_input(how_many = how_many)
    example_outputs = pro_handler.infer(example_inputs)

    t_infer = timer() - t_infer
    times.append(t_infer)
    print("Prediction (of",how_many,"samples) took", t_infer, "sec.")

times = np.asarray(times)
print("Statistics:")
print("prediction of whole",how_many," took time - avg +- std =", np.mean(times), "+-", np.std(times), "sec.")
print("prediction as divided for one - avg +- std =", np.mean(times/how_many), "+-", np.std(times/how_many), "sec.")
