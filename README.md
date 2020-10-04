
# GAN Explorer
 
<p align="center">
<img src="https://github.com/previtus/GAN_explorer/raw/master/illustration/fig2-application.jpg" width="500">
</p>

**Real-time interaction** with deep generative architectures. Load and explore your Progressive GAN models. Enables Convolutional Layer Reconnection, a **novel interaction technique** which disconnects and reconnects convolutional layer connections in the model graph. Finally it’s possible to **save the edited neural network and to reuse it in other Creative AI applications**.

This repository contains the code for real-time interaction, reconnection and also a demo audio-visual application working with LSTM generated music and corresponding GAN generated visuals.

## Convolutional Layer Reconnection

<p align="center">
<img src="https://github.com/previtus/GAN_explorer/raw/master/illustration/fig1-convolutional_layer_reconnection.jpg" width="500">
</p>


Proposed novel mode of interaction with deep generative models by changing the connectivity in the network graph, namely in its convolutional layers (which are likely to be present in any modern generative model and as such this technique can be used universally). Targeting different depths of the generative networks enables different types of resulting effects (these are surprising and innovative yet allow for some degree of predictability). Changes in low-resolution convolutional layers (such as the _16x16_ resolution under tensor _“16x16/Conv0/weight”_) causes changes in the conceptual information of the generated image, while changes in the high-resolution convolutional layers (such as _256x256_) influence details and textures.

## Docker demo

We have a new demo using the nvidia docker at https://hub.docker.com/repository/docker/previtus/demo-gan-explorer.

After installing [docker](https://docs.docker.com/get-docker/) and it's [nvidia drivers](https://github.com/NVIDIA/nvidia-docker), you should be able to easily run our code with:

`sudo docker run -it --rm --user=$(id -u $USER):$(id -g $USER) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --env="QT_X11_NO_MITSHM=1" -v $(pwd)/renders/:/home/GAN_explorer/renders/ --gpus all -ti previtus/demo-gan-explorer`

This docker supports showing the default pre-trained model. Please check the readme for instructions on how to run your own trained Progressive GAN models. Currently was tested on Linux.

## Using GAN Explorer

To use GAN Explorer with your own custom Progressive GAN models, change the network path in the following demo commands (we will otherwise refer to a model released with the original Progressive GAN paper - _"karras2018iclr-lsun-car-256x256.pkl"_).

### Exploring latent space and using Convolutional Layer Reconnection

`python demo.py -network karras2018iclr-lsun-car-256x256.pkl`

**Controls**

Explore the model latent space by game-like controls: `w,s,a,d`. Use `w,s` to move forwards/backwards in one of the 512 latent dimensions (move forward means adding a small number to the vector in the selected dimension in this context), while the `a,d` changes the chosen dimension (we recommend moving forward and backward after each change of dimension to visualize the concept which this dimension represents). Use `r` to jump to a new location in the latent space (and `e` to return to the previous location). `Shift` toggles whether you save or load from the numpad key positions: `1,2,...9`. `=` starts an interpolation animation between the saved latents (note that with `]` you can trigger these interpolations to be rendered into png files). You can also export the saved latents for future reuse with `k` (this saves them into `latents/save`), or load latents from folder with `l` (this loads latents from `latents/load` - this distinction between the folder allows better management of the used latents).

**Convolutional Layer Reconnection controls**

You can trigger the Convolutional Layer Reconnection technique with your currently loaded model at your current latent vector position with `h`. This will alter the current network connectivity graph with a strength which can be controlled in the argument: `-conv_reconnect_str 0.3` (0.3 corresponds to 30% of connections being targeted in the selected depth; repeat this press for a repetition of the effect). By pressing `t`, you can cycle through the targeted convolutional layer depths (see the code output for indication of the selected depth).

<p align="center">
<img src="https://github.com/previtus/GAN_explorer/raw/master/illustration/fig3-plot.jpg" width="650">
</p>

You can use `p` to plot the current latent space position with gradually increasing strength and depth of the Convolutional Layer Reconnection effect (these grid visualizations correspond to the Figures used in our paper).
Finally, you can save your edited Progressive GAN generator models with `u` to allow for usage of this altered model in other Creative AI applications.

### Server deployed version

By default, this code will run with a local Progressive GAN handler, but it can also connect to a server machine. In this way the client side code can run only the image rendering and user interaction part and the server machine with a better performing GPU can handle the Progressive GAN model inference. Note that the current reconnection functionality is limited to a version running without server, however it is possible to deploy the edited networks onto the server machine. Also keep in mind that the client to server communication will possibly slow down the expected performance of the code. 

The client code will try to connect to a RESTful API at `http://localhost:"+PORT+"/get_image"` where the default port of 8000 can be optionally changed. Please use SSH tunneling to enable this routing between the client and the server (see _ssh_tunnel_instructions.txt for more details).

**On server side run:**

`python server.py -network karras2018iclr-lsun-car-256x256.pkl` (optionally -port 8123)

**On client side run:**

`python demo.py -deploy True` (optionally -port 8123)

Note that the client machine does not need to have access to the network .pkl file and simply receives images via REST calls.


### OSC listener / Audio-Visual application demo

`python demo.py -mode listen -network karras2018iclr-lsun-car-256x256.pkl`

You can start the GAN Explorer application in a listener model, in which it will wait for 512 sized OSC messages to guide the latent space traversal. In this way the application can be connected to a music generating LSTM models and visualize an embedding of the generated spectral frames.
You will also need to connect the OSC sending part of the code. This can be simulated by:

```
latent = np.random.randn(512)
from oscpy.client import OSCClient # see https://github.com/kivy/oscpy
import numpy as np
osc = OSCClient('0.0.0.0', 8000)
signal_latent = [float(v) for v in latent]
osc.send_message(b'/send_gan_i', [0, 0] + signal_latent)
```

Alternatively check out our [music_gen_interaction_RTML repo]( https://github.com/previtus/music_gen_interaction_RTML).

---

## Installation

**Prerequisite libraries:**

- (the usual suspects) tensorflow + CUDA (we tested `tensorflow_gpu-1.14.0 + CUDA 10.2` but other versions are also likely to work). Depending on the tf version some tensors might be named differently (we noticed that the naming scheme of the pretrained models had “16x16/Conv0/weight” tensors, which were changed to “16x16/Conv0_up/weight” in our own custom models. This is addressed in our code).
- cv2: `sudo apt-get install python-opencv`
- libraries: `pip install opencv-python requests matplotlib Pillow mock oscpy scikit-image`
- (optional) pygame for midi controller - `python3 -m pip install -U pygame --user` (and on linux I had to also: `sudo apt-get install -y libasound2-plugins`)

**Model data:**

- put your own trained Progressively Growing GAN / StyleGAN2 model (or using the official pretrained ones) in `models/`. For example download [karras2018iclr-lsun-car-256x256.pkl]( https://drive.google.com/drive/folders/15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU).

- download the dnnlib code from the [StyleGAN2 repo](https://github.com/NVlabs/stylegan2/tree/master/dnnlib?version=7d3145d) and place it into the dnnlib folder (this is needed by the loaded ProgressiveGAN and StyleGAN2 models upon loading from .pkl).

## Video/Demo:

<p align="center">
<a href="https://www.youtube.com/watch?v=w24XcLon1Ns" title="Convolutional Layer Reconnection preview"><img src="https://img.youtube.com/vi/w24XcLon1Ns/0.jpg" width="500"></a><br>
<em>Convolutional Layer Reconnection preview video</em>
</p>

## Acknowledgements

This repository is using the trained Progressive Growing GAN models from [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans).
