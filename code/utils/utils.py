# -*- coding: utf-8 -*-

import re
import os
import json
import codecs

'''
Used to check the existence of the path. If the path doesn't
exist, raise error if raise_error is True, or make the path.
'''
def CheckPath(path, raise_error=True):
    if not os.path.exists(path):
        if raise_error:
            print("ERROR : Path %s does not exist!" % path)
            exit(1)
        else:
            print("WARNING : Path %s does not exist!" % path)
            print("INFO : Creating path %s." % path)
            os.makedirs(path)
            print("INFO : Successfully making dir!")
    return

'''
Used to print arguments on the screen.
'''
def printArgs(args):
    print("="*20 + "Arguments" + "="*20)
    argsDict = vars(args)
    for arg, value in argsDict.items():
        print("==> {} : {}".format(arg, value))
    print("="*50)
