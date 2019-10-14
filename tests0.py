import pickle, pickletools
import numpy as np
import tensorflow as tf
import PIL.Image
# This finds the tfutils.py as a file

# Initialize TensorFlow session.
tf.InteractiveSession()
path = "aerials512vectors1024px_snapshot-010200.pkl"
with open(path, 'rb') as file:
    print(file)

    proto_op = next(pickletools.genops(file))
    assert proto_op[0].name == 'PROTO'
    proto_ver = proto_op[1]
    print("Pickled with version", proto_ver)

    G, D, Gs = pickle.load(file)



"""
def save_imageONE_from_latentONE(latents, name="img", idx_over=0):
    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    # Run the generator to produce a set of images.
    images = Gs.run(latents, labels)

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC
"""

N = 6
latent_vec_size = 512
latents = np.random.RandomState(N).randn(N, *Gs.input_shapes[0][1:]) # 1000 random latents
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

print(images.shape)

i = 3
import matplotlib.pyplot as plt
imgplot = plt.imshow(images[3])