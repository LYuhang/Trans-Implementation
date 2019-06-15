# -*- coding: utf-8 -*-

'''
Filename : log.py
Function : Used to record the running logs.
'''

import os
import sys
import time
import logging

'''
Used to record running log, implement as following:
==> Step1: Initialize a logging file with time prefix
==> Step2: Initialize a logger and handlers(fileHandler and streamHandler)
==> Step3: Set the formatter(according to the mode)
==> Step4: Add the handlers
By calling log() method, message can be recorded to the file and printed 
on the screen.Finally, fileHandler and streamHandler should be closed.
'''

class Logging():
    def __init__(self, recPath, recMode='Norm'):
        if not os.path.exists(recPath):
            print("ERROR : Record path does not exist!")
            exit(1)
        self._initFile(recPath)
        self._initLogger()
        self._setMode(recMode)
        self._addHandler()

    def _initFile(self, recPath):
        files = os.listdir(recPath)
        # Delete all existing log files
        for fn in files:
            if os.path.splitext(fn)[-1] == ".log":
                os.remove(os.path.join(recPath, fn))
        t = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
        self.logfilepath = os.path.join(recPath, "{}.log".format(t))

    def _initLogger(self):
        # Init logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.DEBUG)
        self.fileHandler = logging.FileHandler(self.logfilepath, mode='a')
        self.strmHandler = logging.StreamHandler(sys.stdout)

    def _setMode(self, recMode):
        if recMode == "Norm":
            fileFormatter = logging.Formatter('[%(asctime)s]%(levelname)s:%(message)s')
            strmFormatter = logging.Formatter('[%(asctime)s]%(levelname)s:%(message)s')
            self.fileHandler.setFormatter(fileFormatter)
            self.strmHandler.setFormatter(strmFormatter)
        else:
            print("ERROR : The record mode %s is not supported!")
            exit(1)

    def _addHandler(self):
        self.logger.addHandler(self.fileHandler)
        self.logger.addHandler(self.strmHandler)

    def log(self, message):
        self.logger.info(message)

    def close(self):
        self.fileHandler.close()
        self.strmHandler.close()