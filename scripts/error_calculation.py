#!/usr/bin/env python

import time
import scipy.io
import os
import rospy


class ErrorCalculation:
    def __init__(self, hand_parts, frequency):
        self.running = False
        self.hand_parts = hand_parts
        self.frequency = frequency
        self.errors = []
        self.start_time = 0
        self.execution_times = 0
        self.last_hpe_update = 0
        self.per_finger_errors = {}
        for hand_part in self.hand_parts:
            self.per_finger_errors[hand_part.getName()] = []

    def __del__(self):
        self.saveResults()

    def calculateError(self):
        if self.running and self.start_time + 1. / self.frequency * self.execution_times < time.time():
            self.execution_times += 1
            whole_error = 0.
            for hand_part in self.hand_parts:
                this_error = hand_part.getAllTaskDescriptorsErrors()
                whole_error = whole_error + this_error
                self.per_finger_errors[hand_part.getName()].append(this_error)
            self.errors.append(whole_error)
            if self.last_hpe_update + 4. < time.time():
                self.stop()

    def start(self):
        if not self.running:
            self.running = True
            self.start_time = time.time()
            rospy.loginfo("Starting calculation of errors for task descriptors.")

    def stop(self):
        self.running = False
        self.saveResults()
        rospy.loginfo("Finished calculation of errors for task descriptors.")

    def saveResults(self):
        folder_path = 'error_results'
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        file_name = ''
        for hand_part in self.hand_parts:
            file_name += hand_part.getName() + hand_part.taskDescriptorsCount()
        dict_to_save = self.per_finger_errors
        dict_to_save['whole_error'] = self.errors
        scipy.io.savemat(file_name+'.mat', mdict=dict_to_save)

