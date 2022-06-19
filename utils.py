import os
import random

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def get_rand_array(count, max_number):
    return [random.randint(0, max_number-1) for i in range(count)]

