import tensorflow as tf


def get_max(array):
    max = array[0]
    for i in array:
        if i > max:
            max = i
    return max


print(get_max([1, 3, 4, 2, 4, 5, 6, 7, 4, 5]))
