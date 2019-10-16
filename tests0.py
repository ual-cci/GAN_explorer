import pickle, pickletools
import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

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


"""

Gs                          Params      OutputShape             WeightShape             
---                         ---         ---                     ---                      
ToRGB_lod2                  195         (3,)                    (1, 1, 64, 3)           
Upscale2D_5                 -           (?, 3, 256, 256)        -                       
Grow_lod2                   -           (?, 3, 256, 256)        -                       
512x512/Conv0_up            18464       (32,)                   (3, 3, 32, 64)          
512x512/Conv1               9248        (32,)                   (3, 3, 32, 32)          
ToRGB_lod1                  99          (3,)                    (1, 1, 32, 3)           
Upscale2D_6                 -           (?, 3, 512, 512)        -                       
?? Grow_lod1                   -           (?, 3, 512, 512)        -                       
1024x1024/Conv0_up          4624        (16,)                   (3, 3, 16, 32)          
1024x1024/Conv1             2320        (16,)                   (3, 3, 16, 16)          
ToRGB_lod0                  51          (3,)                    (1, 1, 16, 3)           
Upscale2D_7                 -           (?, 3, 1024, 1024)      -                       
?? Grow_lod0                   -           (?, 3, 1024, 1024)      -                       
?? images_out                  -           (?, 3, 1024, 1024)      -                       
---                         ---         ---                     ---                     
Total                       23079115                                                    


"""

#Gs.print_layers()

#ls = Gs.list_layers()
#for l in ls:
#    print(l)

"""
('512x512/Conv0_up', <tf.Tensor 'Gs/512x512/Conv0_up/bias/setter:0' shape=(32,) dtype=float32_ref>, [<tf.Tensor 'Gs/512x512/Conv0_up/weight:0' shape=(3, 3, 32, 64) dtype=float32_ref>, <tf.Tensor 'Gs/512x512/Conv0_up/bias:0' shape=(32,) dtype=float32_ref>])
('512x512/Conv1', <tf.Tensor 'Gs/512x512/Conv1/bias/setter:0' shape=(32,) dtype=float32_ref>, [<tf.Tensor 'Gs/512x512/Conv1/weight:0' shape=(3, 3, 32, 32) dtype=float32_ref>, <tf.Tensor 'Gs/512x512/Conv1/bias:0' shape=(32,) dtype=float32_ref>])
('ToRGB_lod1', <tf.Tensor 'Gs/ToRGB_lod1/bias/setter:0' shape=(3,) dtype=float32_ref>, [<tf.Tensor 'Gs/ToRGB_lod1/weight:0' shape=(1, 1, 32, 3) dtype=float32_ref>, <tf.Tensor 'Gs/ToRGB_lod1/bias:0' shape=(3,) dtype=float32_ref>])
('Upscale2D_6', <tf.Tensor 'Gs/Upscale2D_6/Reshape_1:0' shape=(?, 3, 512, 512) dtype=float32>, [])
<<<?<<< ('Grow_lod1', <tf.Tensor 'Gs/Grow_lod1/add:0' shape=(?, 3, 512, 512) dtype=float32>, [])
('1024x1024/Conv0_up', <tf.Tensor 'Gs/1024x1024/Conv0_up/bias/setter:0' shape=(16,) dtype=float32_ref>, [<tf.Tensor 'Gs/1024x1024/Conv0_up/weight:0' shape=(3, 3, 16, 32) dtype=float32_ref>, <tf.Tensor 'Gs/1024x1024/Conv0_up/bias:0' shape=(16,) dtype=float32_ref>])
('1024x1024/Conv1', <tf.Tensor 'Gs/1024x1024/Conv1/bias/setter:0' shape=(16,) dtype=float32_ref>, [<tf.Tensor 'Gs/1024x1024/Conv1/weight:0' shape=(3, 3, 16, 16) dtype=float32_ref>, <tf.Tensor 'Gs/1024x1024/Conv1/bias:0' shape=(16,) dtype=float32_ref>])
('ToRGB_lod0', <tf.Tensor 'Gs/ToRGB_lod0/bias/setter:0' shape=(3,) dtype=float32_ref>, [<tf.Tensor 'Gs/ToRGB_lod0/weight:0' shape=(1, 1, 16, 3) dtype=float32_ref>, <tf.Tensor 'Gs/ToRGB_lod0/bias:0' shape=(3,) dtype=float32_ref>])
<<< ? <<< ('Upscale2D_7', <tf.Tensor 'Gs/Upscale2D_7/Reshape_1:0' shape=(?, 3, 1024, 1024) dtype=float32>, [])
<<< ? <<< ('Grow_lod0', <tf.Tensor 'Gs/Grow_lod0/add:0' shape=(?, 3, 1024, 1024) dtype=float32>, [])
('images_out', <tf.Tensor 'Gs/images_out:0' shape=(?, 3, 1024, 1024) dtype=float32>, [])

check images_out vs Grow_lod0 (i guess)

"""



latent_vec_size = 512


# Run the generator to produce a set of images.

output_layer_name = "Gs/Grow_lod0/add:0" # just like original
output_layer_name = "Gs/images_out:0" # original
output_layer_name = "Gs/Upscale2D_7/Reshape_1:0" # same res, diff colors (just swapped maybe)
#output_layer_name = "Gs/Grow_lod1/add:0" # same diff in colors, 512x512 tho!
#output_layer_name = "Gs/Grow_lod3/add:0" #

names_to_try = ["Gs/Grow_lod0/add:0", "Gs/images_out:0", "Gs/Upscale2D_7/Reshape_1:0"]


names_to_try = []
for i in range(0,6):
    names_to_try.append("Gs/Grow_lod"+str(i)+"/add:0")

from PIL import Image


for output_layer_name in names_to_try:
    restart = True
    for repeat_i in range(10):
        latents = np.random.RandomState(repeat_i).randn(1, *Gs.input_shapes[0][1:])  # 1000 random latents
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

        t_infer = timer()

        images = Gs.custom_tinkered_layer_output_run(output_layer_name, restart,latents, labels)
        restart = False
        print(images.shape)
        #print(images)
        #images = Gs.run(latents, labels)
        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

        t_infer = timer() - t_infer
        print("Prediction+process of ",len(images)," took", t_infer, "sec.")


        print(images.shape)
        im = Image.fromarray(images[0])
        filename = output_layer_name.replace("/","_").replace(":","-") + "___repeat"+str(repeat_i)+".jpg"
        #im.save(filename)

"""
import matplotlib.pyplot as plt
imgplot = plt.imshow(images[i])

plt.show()
"""