import numpy as np
from settings import Settings
from timeit import default_timer as timer

# Blend two generative NNs
second_net = None

from dnnlib.tflib import tfutil

def load_nets(first_net_path = 'tempweights-walk.npy', second_net_weights='tempweights.npy'):
    global weights, weightsfirstnet
    weights = np.load(second_net_weights, allow_pickle=True).item()
    weightsfirstnet = np.load(first_net_path, allow_pickle=True).item()

load_nets('tempweights-walk.npy', 'tempweights.npy')
#weights = np.load('tempweights.npy', allow_pickle=True).item()
#weightsfirstnet = np.load('tempweights-walk.npy', allow_pickle=True).item()
#print("reporting loaded weights:", weights.keys())

def slow_blend_from_saved_weights(net, alpha=0.9, verbose=False):
    if verbose: print("first net.vars", net.vars)
    #print("second net.vars", second_net.vars)

    for tensor_key in net.vars:
        blended_dicts = {}

        #if "G_synthesis" in tensor_key and (("_up" not in tensor_key) or ("ToRGB" in tensor_key)):
        #if "G_synthesis" in tensor_key and "1024x1024" in tensor_key:
        if "G_synthesis" in tensor_key:
            # so far it seems like for StyleGan2 we ...
            # - need ToRGB, need both the ones without _up and with _up
            try:
                if verbose: print(tensor_key)

                first_net_weights = weightsfirstnet[tensor_key] # net.get_var(tensor_key)
                second_net_weights = weights[tensor_key] # second_net.get_var(tensor_key)

                blended_weights = (1.0-alpha) * first_net_weights + (alpha) * second_net_weights
                blended_weights = np.copy(blended_weights)

                #net.set_var(tensor_key, np.copy( blended_weights ))
                v = net.find_var(tensor_key)
                tfutil.set_vars({v: blended_weights})

                #blended_dicts[tensor_key] = np.copy( blended_weights )

                """
                start = timer()
                end = timer()
                time = (end - start)
                print("Save to net " + str(time) + "s")
                """
            except Exception as e:
                print("--failed on tensor", tensor_key, "with:", e)

    """
    blended_dicts_tmp = {}
    for key, value in blended_dicts:
        k = net.find_var(key)
        blended_dicts_tmp[k] = value
        tfutil.set_vars({k: value}) # < try without too
    #tfutil.set_vars(blended_dicts_tmp)
    """

    return net


"""
Load from net 0.156369910997455s
Load from dictionary 1.5929981600493193e-06s
Save to net 0.15563486399696558s
"""

import argparse
parser = argparse.ArgumentParser(description='Project: GAN Explorer - weights prep.')
parser.add_argument('-network', help='Path to the network to export weights from.', default='network-snapshot-000491.pkl')
parser.add_argument('-architecture', help='"ProgressiveGAN" or "StyleGAN2".', default="StyleGAN2")


if __name__ == '__main__':
    args_main = parser.parse_args()

    #### LOAD A SECOND NET:
    import mock
    import os

    args = mock.Mock()
    args.architecture = args_main.architecture
    args.model_path = args_main.network
    print(" ... loading from ... ", args.model_path)

    settings = Settings()

    serverside_handler = None

    if args.architecture == "ProgressiveGAN":
        import progressive_gan_handler
        serverside_handler = progressive_gan_handler.ProgressiveGAN_Handler(settings, args)
    if args.architecture == "StyleGAN2":
        import stylegan2_handler
        serverside_handler = stylegan2_handler.StyleGAN2_Handler(settings, args)

    loaded_net = serverside_handler._Gs
    print("reporting locally loaded net:", loaded_net)

    weights = {}

    for tensor_key in loaded_net.vars:
        try:
            first_net_weights = loaded_net.get_var(tensor_key)
            weights[ tensor_key ] = first_net_weights
        except Exception as e:
            print("--failed on tensor", tensor_key, "with:", e)

    base = os.path.basename(args.model_path)
    save_name = os.path.dirname(args.model_path) + "/" + os.path.splitext(base)[0] + ".npy"
    print("... saving to ... ", save_name)
    np.save(save_name, weights)


#python blender.py -network /media/vitek/4E3EC8833EC86595/Vitek/ResearchProjectsWork/DownloadsFromARC/stylegan2/walk-network-snapshot-001309.pkl