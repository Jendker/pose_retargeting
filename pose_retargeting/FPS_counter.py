#!/usr/bin/env python

import time
import numpy as np


class FPSCounter:
    def __init__(self):
        current_time = time.time()
        self.start_time_for_display = current_time
        self.last_time = current_time
        self.x = 5  # displays the frame rate every X second
        self.time_between_calls = []
        self.elements_for_mean = 20

    def getAndPrintFPS(self):
        current_time = time.time()
        self.time_between_calls.append(1.0/(current_time - self.last_time))
        if len(self.time_between_calls) > self.elements_for_mean:
            self.time_between_calls.pop(0)
        self.last_time = current_time
        frequency = np.mean(self.time_between_calls)
        if (current_time - self.start_time_for_display) > self.x:
            print("Frequency: {}Hz".format(int(frequency)))
            self.start_time_for_display = current_time
        return frequency
