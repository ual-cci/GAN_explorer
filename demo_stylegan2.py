# DEMO to run


from getter_functions import Getter
from interaction_handler import Interaction_Handler

import mock

args = mock.Mock()
args.architecture = "StyleGAN2"
args.model_path = "../stylegan2/stylegan2-ffhq-config-f.pkl"
#args.model_path = "models/stylegan2models/network-snapshot-007782.pkl"
#args.model_path = "models/stylegan2models/network-snapshot-006512.pkl"
args.model_path = "models/stylegan2models/network-snapshot-005243.pkl"

getter = Getter(args)

interaction_handler = Interaction_Handler(getter)
interaction_handler.latent_vector_size = getter.get_vec_size_localServerSwitch()

version = "v0" # random
version = "v0b" # random + interpolation
version = "v2" # "game"

steps_speed = 120

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
