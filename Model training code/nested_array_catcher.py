import numpy as np


def nested_array_catcher(array, target_type=np.float32):
    for i in range(len(array)):
        if isinstance(array[i], target_type):
            continue
        else:
            print('Found a nested array')
            print(array.shape, type(array), type(array[i])) # <class 'numpy.ndarray'>
            print(array, array[i])
            array = array[i]
            print('new type: ', array.shape, type(array), type(array[i]))
            print(array, array[i])
            return array
    return array
