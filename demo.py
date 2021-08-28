# DEMO to run


from getter_functions import Getter
from interaction_handler import Interaction_Handler

import argparse

parser = argparse.ArgumentParser(description='Project: GAN Explorer.')

# python demo.py -h -> prints out help
parser.add_argument('-mode', help='Mode under which we run GAN Explorer ("explore" - explore the latent space and use techniques such as Convolutional Layer Reconnection / "listen" - OSC signal listener mode, audio-visual demo (see the readme)). Defaults to "explore".', default='explore')
parser.add_argument('-network', help='Path to the model (.pkl file) - this can be a pretrained ProgressiveGAN model, or just the Generator network (Gs).', default='models/karras2018iclr-lsun-car-256x256.pkl')
parser.add_argument('-architecture', help='GAN architecture type (support for "ProgressiveGAN"; work-in-progress also "StyleGAN2"). Defaults to "ProgressiveGAN".', default='ProgressiveGAN')
parser.add_argument('-steps_speed', help='Interpolation speed - steps_speed controls how many steps each transition between two samples will have (large number => smoother interpolation, slower run). Suggested 60 (mid-end) or 120 (high-end). Defaults to 60.', default='60')
parser.add_argument('-conv_reconnect_str', help='Strength of one Convolutional Layer Reconnection effect (0.3 defaults to 30 percent of the connections being reconnected in each click).', default='0.3')

parser.add_argument('-deploy', help='Optional mode to depend on a deployed run of the Server.py code (see python server.py -h for more).', default='False')
parser.add_argument('-port', help='Server runs on this port. Defaults to 8000 (this uses the link "http://localhost:"+PORT+"/get_image" for rest calls. Use SSH tunel.', default='8000')

parser.add_argument('-fullscreen', help='Start Fullscreen? ("full" - fullscreen, "resize" - resizeable window, otherwise fixed resolution 1024)', default='None')
parser.add_argument('-skip_intro', help='Directly into game?.', default='False')


if __name__ == '__main__':
    args_main = parser.parse_args()

    import mock
    args = mock.Mock()
    args.architecture = str(args_main.architecture)

    import os
    # if there is another model in the fixed path of networks/net.pkl, use that one instead:
    if os.path.exists("networks/net.pkl"):
        args.model_path = "networks/net.pkl"
    else:
        args.model_path = str(args_main.network)

    #####################
    # Local override
    #args.model_path = "models/grayjungledwellers-008800.pkl"


    steps_speed = int(args_main.steps_speed)

    #version = "v0" # random
    #version = "v0b" # random + interpolation
    version = "v2"  # "game"
    mode = str(args_main.mode)
    if mode == "explore":
        version = "v2" # "game"
    elif mode == "listen":
        version = "v1"  # OSC listener

    server_deployed = (args_main.deploy == "True")
    port = str(args_main.port) #port = "8000" # -> Uses a link for REST requests: "http://localhost:"+PORT+"/get_image"
    getter = Getter(args, USE_SERVER_INSTEAD=server_deployed, PORT=port)
    initial_resolution = 1024
    fullscreen = str(args_main.fullscreen)
    skip_intro = (args_main.skip_intro == "True")

    interaction_handler = Interaction_Handler(getter, initial_resolution, fullscreen, start_in_autonomous_mode=skip_intro)
    interaction_handler.convolutional_layer_reconnection_strength = float(args_main.conv_reconnect_str)

    pretrained_model = ("karras2018iclr" in args.model_path)
    if args.architecture == "ProgressiveGAN":
        if not pretrained_model:
            # << Pre-trained PGGAN models have tensors named as: "16x16/Conv0/weight" while our custom models have "16x16/Conv0_up/weight" -> probably due to the used tf versions
            interaction_handler.target_tensors = [tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.target_tensors]
            interaction_handler.plotter.target_tensors = [tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.plotter.target_tensors]
        if "-256x256.pkl" in args.model_path:
            interaction_handler.plotter.font_multiplier = 0.25
    ### StyleGAN2 layer naming is different:
    if args.architecture == "StyleGAN2":
        interaction_handler.target_tensors = ["G_synthesis/"+tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.target_tensors]
        interaction_handler.plotter.target_tensors = ["G_synthesis/"+tensor.replace("Conv0", "Conv0_up") for tensor in interaction_handler.plotter.target_tensors]


    # plotter allowed only in local run
    if not server_deployed:
        interaction_handler.plotter.prepare_with_set_tensors()

    interaction_handler.latent_vector_size = getter.get_vec_size_localServerSwitch()

    if version == "v0":
        interaction_handler.start_renderer_no_interaction()

    elif version == "v0b":
        interaction_handler.shuffle_random_points(steps=steps_speed)
        interaction_handler.keep_p1 = True # << optional
        interaction_handler.start_renderer_interpolation()

    elif version == "v1":

        interaction_handler.shuffle_random_points(steps=steps_speed)
        interaction_handler.start_renderer_interpolation_interact()

    elif version == "v2":
        interaction_handler.shuffle_random_points(steps=steps_speed)
        interaction_handler.start_renderer_key_interact(skip_intro=skip_intro)
