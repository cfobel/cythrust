import timeit

import numpy as np


def auto_timeit(statement, setup, number=0):
    timer = timeit.Timer(statement, setup)
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        number = 1
        for _ in range(1, 10):
            runtime = timer.timeit(number=number)
            if runtime >= 0.2:
                break
            number *= 10
    return np.array(timer.repeat(number=number)) / number
