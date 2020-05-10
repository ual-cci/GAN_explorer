# DEMO to run


from getter_functions import Getter
from interaction_handler import Interaction_Handler



import mock

args = mock.Mock()
args.architecture = "ProgressiveGAN"
args.model_path = 'models/karras2018iclr-celebahq-1024x1024.pkl' # colors shifted ...
args.model_path = 'models/karras2018iclr-lsun-airplane-256x256.pkl'

args.model_path = 'models/aerials128vectors256px_-snapshot-007440.pkl'  # 50fps ==> still 50fps
args.model_path = 'models/aerials512vectors1024px_snapshot-010200.pkl' # 20fps ==> 15fps
args.model_path = 'models/grayjungledwellers-008400.pkl'               # 33fps (BW) ==> 28fps


args.model_path = 'models/karras2018iclr-celebahq-1024x1024.pkl'
getter = Getter(args)
initial_resolution = 1024

interaction_handler = Interaction_Handler(getter, initial_resolution)
interaction_handler.latent_vector_size = getter.get_vec_size_localServerSwitch()

version = "v0" # random
version = "v0b" # random + interpolation
version = "v2" # "game"

#version = "v1" # OSC listener
steps_speed = 120

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
    interaction_handler.start_renderer_key_interact()
