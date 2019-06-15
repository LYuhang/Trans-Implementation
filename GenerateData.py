# -*- coding:utf-8 -*-
'''
Filename : GenerateData.py
'''

from code.process.triples import csvToStandard, jsonToStandard, \
                                generateDict, splitData

if __name__ == "__main__":
    trainFile = "./data/freebase_mtr100_mte100-train.txt"
    validFile = "./data/freebase_mtr100_mte100-valid.txt"
    testFile = "./data/freebase_mtr100_mte100-test.txt"
    saveDir = "./data/"
    dictDir = "./source/dict/"
    # Step1: Transform raw data to standard format
    csvToStandard(rawPath=trainFile,
                  savePath="./data/train.txt",
                  names=["head", "relation", "tail"],
                  header=None,
                  sep="\t",
                  encoding="utf-8")
    csvToStandard(rawPath=validFile,
                  savePath="./data/valid.txt",
                  names=["head", "relation", "tail"],
                  header=None,
                  sep="\t",
                  encoding="utf-8")
    csvToStandard(rawPath=testFile,
                  savePath="./data/test.txt",
                  names=["head", "relation", "tail"],
                  header=None,
                  sep="\t",
                  encoding="utf-8")

    # Step2: Generate dict
    generateDict(dataPath=[trainFile, validFile, testFile],
                 dictSaveDir=dictDir)

    # Step3: Split data
    pass