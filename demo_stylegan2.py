# DEMO to run


from getter_functions import Getter
from interaction_handler import Interaction_Handler
import argparse

parser = argparse.ArgumentParser(description='Project: GAN Explorer.')

# python demo.py -h -> prints out help
parser.add_argument('-mode', help='Mode under which we run GAN Explorer ("explore" - explore the latent space and use techniques such as Convolutional Layer Reconnection / "listen" - OSC signal listener mode, audio-visual demo (see the readme)). Defaults to "explore".', default='explore')
parser.add_argument('-network', help='Path to the model (.pkl file) - this can be a pretrained ProgressiveGAN model, or just the Generator network (Gs).', default='models/stylegan2/stylegan2-ffhq-config-f.pkl')
parser.add_argument('-architecture', help='GAN architecture type (support for "ProgressiveGAN"; work-in-progress also "StyleGAN2"). Defaults to "ProgressiveGAN".', default='StyleGAN2')
parser.add_argument('-steps_speed', help='Interpolation speed - steps_speed controls how many steps each transition between two samples will have (large number => smoother interpolation, slower run). Suggested 60 (mid-end) or 120 (high-end). Defaults to 60.', default='120')
parser.add_argument('-conv_reconnect_str', help='Strength of one Convolutional Layer Reconnection effect (0.3 defaults to 30 percent of the connections being reconnected in each click).', default='0.3')

parser.add_argument('-deploy', help='Optional mode to depend on a deployed run of the Server.py code (see python server.py -h for more).', default='False')
parser.add_argument('-port', help='Server runs on this port. Defaults to 8000 (this uses the link "http://localhost:"+PORT+"/get_image" for rest calls. Use SSH tunel.', default='8000')

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

    args.architecture = "StyleGAN2"
    args.model_path = "/media/vitek/4E3EC8833EC86595/Vitek/ResearchProjectsWork/DownloadsFromARC/stylegan2/00009-stylegan2-bus_35k_1024-1gpu-config-e/network-snapshot-000491.pkl"

    steps_speed = int(args_main.steps_speed)

    #version = "v0" # random
    #version = "v0b" # random + interpolation
    version = "v2"  # "game"
    mode = str(args_main.mode)
    if mode == "explore":
        version = "v2" # "game"
    elif mode == "listen":
        version = "v1"  # OSC listener

    getter = Getter(args)

    interaction_handler = Interaction_Handler(getter)
    interaction_handler.latent_vector_size = getter.get_vec_size_localServerSwitch()

    if version == "v0":
        interaction_handler.start_renderer_no_interaction()

    elif version == "v0b":
        interaction_handler.shuffle_random_points(steps=steps_speed)
        interaction_handler.keep_p1 = True # << optional
        interaction_handler.start_renderer_interpolation()

    elif version == "v1":
        OSC_address = '0.0.0.0'
        OSC_port = 8000
        OSC_bind = b'/send_gan_i'

        SIGNAL_interactive_i = 0.0
        SIGNAL_reset_toggle = 0

        # OSC - Interactive listener
        def callback(*values):
            global SIGNAL_interactive_i
            global SIGNAL_reset_toggle
            print("OSC got values: {}".format(values))
            # [percentage, model_i, song_i]
            percentage, reset_toggle = values

            SIGNAL_interactive_i = float(percentage) / 1000.0  # 1000 = 100% = 1.0
            SIGNAL_reset_toggle = int(reset_toggle)

        print("Also starting a OSC listener at ", OSC_address, OSC_port, OSC_bind, "to listen for interactive signal (0-1000).")
        from oscpy.server import OSCThreadServer
        osc = OSCThreadServer()
        sock = osc.listen(address=OSC_address, port=OSC_port, default=True)
        osc.bind(OSC_bind, callback)

        interaction_handler.shuffle_random_points(steps=steps_speed)
        interaction_handler.start_renderer_interpolation_interact()

    elif version == "v2":
        interaction_handler.shuffle_random_points(steps=steps_speed)
        interaction_handler.start_renderer_key_interact()
