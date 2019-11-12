# DEMO to run


from getter_functions import Getter
from interaction_handler import Interaction_Handler



import mock

args = mock.Mock()

args.model_path = 'models/karras2018iclr-celebahq-1024x1024.pkl' # colors shifted ...
args.model_path = 'models/karras2018iclr-lsun-airplane-256x256.pkl'

args.model_path = 'models/aerials128vectors256px_-snapshot-007440.pkl'  # 50fps ==> still 50fps
args.model_path = 'models/aerials512vectors1024px_snapshot-010200.pkl' # 20fps ==> 15fps
args.model_path = 'models/grayjungledwellers-008400.pkl'               # 33fps (BW) ==> 28fps

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
