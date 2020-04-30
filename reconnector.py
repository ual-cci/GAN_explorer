import numpy as np

# Reconnect convolution blocks inside NN:


# helper func to swap
def swap(weights, fixed, num1, num2):
    # w[:,:,fixed, num1] <=> w[:,:,fixed, num2]
    conv1vals = weights[:, :, fixed, num1]
    # print("conv1vals.shape", conv1vals.shape)
    conv2vals = weights[:, :, fixed, num2]
    # print("conv2vals.shape", conv2vals.shape)

    weights[:, :, fixed, num2] = conv1vals
    weights[:, :, fixed, num1] = conv2vals
    return weights


# reconnection of conv filters in existing net
def reconnect(net, tensor_name="128x128/Conv0_up/weight", percent_change=10, DO_ALL=True):
    weights = get_tensor(net, tensor_name)

    print("weights.shape", weights.shape)
    res = weights.shape[2]
    possible = list(range(res))
    to_select = int((res / 100.0) * percent_change)

    select = np.random.choice(possible, to_select, replace=False)

    odds = []
    evens = []
    for i in range(len(select)):
        if i % 2 == 0:
            evens.append(i)
        else:
            odds.append(i)

    # print(select)
    # print(odds)
    # print(evens)

    equalizer = min(len(odds), len(evens))
    evens = evens[0:equalizer]
    odds = odds[0:equalizer]

    AS = select[odds]
    BS = select[evens]

    # print(AS)
    # print(BS)

    if DO_ALL:
        NUMS = list(range(res))
    for first in NUMS:
        for idx in range(len(AS)):
            weights = swap(weights, first, AS[idx], BS[idx])

    set_tensor(net, tensor_name, weights)
    return net


def get_tensor_OVERRIDE(net, target_tensor):
    np_arr = net.get_var(target_tensor)
    return np_arr

original_weights_reconnect_specific = {}
def get_tensor(net, target_tensor):
    global original_weights_reconnect_specific
    # first restore net
    for tensor_key in original_weights_reconnect_specific.keys():
        orig_val = original_weights_reconnect_specific[tensor_key]
        net.set_var(tensor_key, orig_val)

    if target_tensor not in original_weights_reconnect_specific:
        # first time getting it
        np_arr = net.get_var(target_tensor)
        original_weights_reconnect_specific[target_tensor] = np_arr
    else:
        np_arr = original_weights_reconnect_specific[target_tensor]

    return np_arr

def set_tensor(net, target_tensor, np_arr):
    net.set_var(target_tensor, np_arr)
    return net

# editednet = reconnect(changedGs, target_tensor, percent_change)
