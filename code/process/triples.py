# -*- coding: utf-8 -*-
'''
Filename : triples.py
Function : Preprocess source data and transform to standard fotmat
==> Generally, there are several steps as following:
==> Step1: Transform raw data to standard format, some supported format lists as following:
==>        1.CSV, TXT, TAR, ZIP: Use pandas to read(pd.read_csv())
==>        2.JSON: Use json to load and transform to standard format
==>        When dealing with the raw data, several exceptions should be considered:
==>        1. There exists blank lines, inconsistent columns(!=3)
==>        2. Head, relation, tail is nan or string that only contains space
==>        3. Some string will be parsed as nan, exp.NAN, N/A, NA and so on
==> Step2: Use data of standard format to generate entity and relation dict
==> Step3: Split train, evaluation and test data if necessary.
'''

import re
import os
import json
import codecs
import numpy as np
import pandas as pd
from collections import Counter

def csvToStandard(rawPath, savePath="./data/standard.txt", names=None, header=None, sep="\t", encoding="utf-8", compression="infer"):
    print("INFO : Loading data of type %s" % os.path.splitext(rawPath)[-1])
    rawDf = pd.read_csv(rawPath,
                        sep=sep,
                        encoding=encoding,
                        names=names,
                        header=header,
                        keep_default_na=False,  # ==> Solve default nan
                        compression=compression,# ==> Solve ZIP and TAR
                        error_bad_lines=False,  # ==> Solve inconsistent lines
                        warn_bad_lines=False,   # ==> Solve inconsistent lines
                        skip_blank_lines=True)  # ==> Solve blank lines
    print("INFO : Remove the space from the head and tail of entity.")
    rawDf = rawDf.applymap(lambda x: x.strip())  # Rid of the space in head and tail of entity
    print("INFO : Drop line with nan value.")    # Attention: " " should be removed.
    rawDf.replace({'': np.nan}, inplace=True)
    rawDf.dropna(axis=0, how='any', inplace=True)

    print("INFO : Save standard data to file path : %s" % savePath)
    rawDf.to_csv(savePath, sep="\t", header=None, index=None, encoding="utf-8")
    print("INFO : Successfully saving!")

def jsonToStandard(jsonPaths, savePath="./data/standard.txt"):
    if type(jsonPaths) == str:
        jsonPaths = [jsonPaths]
    elif type(jsonPaths) == list:
        jsonPaths = jsonPaths
    else:
        print("ERROR : jsonPaths type {} is not supported!".format(type(jsonPaths)))
        exit(1)

    # Read the json file
    J = []
    def dictToStr(d):
        return "{}\t{}\t{}".format(d["head"], d["relation"], d["tail"])
    for p in jsonPaths:
        for d in json.load(codecs.open(p, "r", encoding="utf-8")):
            if re.match("^\s*$", d["head"]) or re.match("^\s*$", d["relation"]) or re.match("^\s*$", d["tail"]):
                continue
            J.append(dictToStr(d))

    # Save as txt file
    print("INFO : Saving txt file!")
    with codecs.open(savePath, "w", encoding="utf-8") as fp:
        fp.write("\n".join(J))

def generateDict(dataPath, dictSaveDir):
    if type(dataPath) == str:
        print("INFO : Loading standard data!")
        rawDf = pd.read_csv(dataPath,
                            sep="\t",
                            header=None,
                            names=["head", "relation", "tail"],
                            keep_default_na=False,
                            encoding="utf-8")
    elif type(dataPath) == list:
        print("INFO : Loading a list of standard data!")
        rawDf = pd.concat([pd.read_csv(p,
                                       sep="\t",
                                       header=None,
                                       names=["head", "relation", "tail"],
                                       keep_default_na=False,
                                       encoding="utf-8") for p in dataPath], axis=0)
        rawDf.reset_index(drop=True, inplace=True)

    headCounter = Counter(rawDf["head"])
    tailCounter = Counter(rawDf["tail"])
    relaCounter = Counter(rawDf["relation"])

    # Generate entity and relation list
    entityList = list((headCounter + tailCounter).keys())
    relaList = list(relaCounter.keys())

    # Transform to index dict
    print("INFO : Transform to index dict")
    entityDict = dict([(word, ind) for ind, word in enumerate(entityList)])
    relaDict = dict([(word, ind) for ind, word in enumerate(relaList)])

    # Save path
    entityDictPath = os.path.join(dictSaveDir, "entityDict.json")
    relaDictPath = os.path.join(dictSaveDir, "relationDict.json")

    # Saving dicts
    json.dump({"stoi": entityDict, "itos": entityList}, open(entityDictPath, "w"))
    json.dump({"stoi": relaDict, 'itos': relaList}, open(relaDictPath, "w"))

def splitData(dataPath, saveDir, evalPortion=0.1):
    assert evalPortion >= 0 and evalPortion <= 1.0
    print("INFO : Loading standard data!")
    rawDf = pd.read_csv(dataPath,
                        sep="\t",
                        header=None,
                        names=["head", "relation", "tail"],
                        keep_default_na=False,
                        encoding="utf-8")
    # Split eval data
    evalDf = rawDf.sample(frac=evalPortion)
    rawDf.drop(labels=evalDf.index, axis=0, inplace=True)
    evalDf.reset_index(drop=True, inplace=True)
    rawDf.reset_index(drop=True, inplace=True)

    # Save path
    trainPath = os.path.join(saveDir, "train.txt")
    validPath = os.path.join(saveDir, "valid.txt")
    rawDf.to_csv(trainPath, sep="\t", header=None, index=None, encoding="utf-8")
    evalDf.to_csv(validPath, sep="\t", header=None, index=None, encoding="utf-8")

'''
This class is suitable for big data and this class just process csv data
and only generates entity dict and relation dict.This will not generate 
negtive samples. 
'''
class BigTripleCorpus():
    def __init__(self, rawPath):
        if not os.path.exists(rawPath):
            print("ERROR : Path %s does not exists!"%rawPath)
            exit(1)

        self.rawPath = rawPath

    def loadBigData(self, chunksize, names, sep=',', header=None, encoding="utf-8", compression="infer"):
        '''
        Load data of big-size
        :return: None
        '''
        print("INFO : Loading big raw data.")
        self.rawDf = pd.read_csv(self.rawPath,
                                 sep=sep,
                                 encoding=encoding,
                                 names=names,
                                 header=header,
                                 chunksize=chunksize,
                                 error_bad_lines=False,
                                 warn_bad_lines=False,
                                 compression=compression)
        print("INFO : Finish loading.")

    def generateDict(self, dictSavePath):
        '''
        Generate entity dict and relation dict
        :param dictSavePath: (str) Entity and relation save dirname
        :return: None
        '''
        if not os.path.exists(dictSavePath):
            print("ERROR : Path %s does not exist!" % dictSavePath)
            exit(1)
        self.dictSavePath = dictSavePath
        headCounter, relaCounter, tailCounter = self._getCounter()

        # Transform to index dict
        print("INFO : Transform to index dict")
        print("INFO : Get entity list.")
        entityList = list((headCounter + tailCounter).keys())
        relaList = list(relaCounter.keys())
        print("==> Entity sum: %d, relation sum: %d" % (len(entityList), len(relaList)))
        print("INFO : Get entity dict.")
        entityDict = dict([(word, ind) for ind, word in enumerate(entityList)])
        relaDict = dict([(word, ind) for ind, word in enumerate(relaList)])

        # Save path
        entityDictPath = os.path.join(self.dictSavePath, "entityDict.json")
        relaDictPath = os.path.join(self.dictSavePath, "relationDict.json")

        # Saving dicts
        print("INFO : Save dict and list as json.")
        json.dump({"stoi": entityDict, "itos": entityList}, open(entityDictPath, "w"))
        json.dump({"stoi": relaDict, 'itos': relaList}, open(relaDictPath, "w"))

    def _getCounter(self, numWorkers=4):
        # Initialize the Counter()
        headCounter = Counter()
        tailCounter = Counter()
        relaCounter = Counter()

        chunkSum = 0
        for chunck in self.rawDf:
            headCounter += Counter(chunck.iloc[:, 0])
            relaCounter += Counter(chunck.iloc[:, 1])
            tailCounter += Counter(chunck.iloc[:, 2])
            chunkSum += len(chunck)
            print("INFO : Process sample num : %d" % chunkSum)

        return headCounter, relaCounter, tailCounter