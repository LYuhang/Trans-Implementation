# -*- coding: utf-8 -*-
'''
Filename : dataloader.py
Usage :
'''

import math
import json
import numpy as np
import pandas as pd
from torch.utils.data import *

class tripleDataset(Dataset):
    def __init__(self, entityDictPath, relationDictPath, posDataPath):
        super(Dataset, self).__init__()
        # Load entity-index dict and relation-index dict
        print("INFO : Load entity and relation dict.")
        self.entityDict = json.load(open(entityDictPath, "r"))["stoi"]
        self.relationDict = json.load(open(relationDictPath, "r"))["stoi"]

        # Transform entity and relation to index
        print("INFO : Loading positive triples and transform to index.")
        self.posDf = pd.read_csv(posDataPath,
                                 sep="\t",
                                 names=["head", "relation", "tail"],
                                 header=None,
                                 encoding="utf-8",
                                 keep_default_na=False)
        '''
        Check the entity and relation.
        '''
        '''
        print(posDataPath)
        print(len(self.posDf))
        for ind in self.posDf.index:
            h, r, t = self.posDf.loc[ind, :]
            if h not in self.entityDict:
                print("Index %d, Head %s not in entityDict!"%(ind, h))
            if t not in self.entityDict:
                print("Index %d, Tail %s not in entityDict!"%(ind, t))
            if r not in self.relationDict:
                print("Index %d, Relation %s not in relationDict!"%(ind, r))
        '''

        self.transformToIndex(self.posDf, repDict={"head":self.entityDict,
                                                   "relation":self.relationDict,
                                                   "tail":self.entityDict})

    '''
    Used to generate negtive sample for training, the params list as following:
    ==> repProba : (float)Probability of replacing head
    ==> exProba : (float)Probability of replacing head with tail entities or replacing 
                  tail with head entities.
    ==> repSeed : Random seed of head replacing probability distribution
    ==> exSeed : Random seed of probability distribution of replacing head with tail 
                 entities or replacing tail with head entities.
    ==> headSeed : Random seed of head shuffling.
    ==> tailSeed : Random seed of tail shuffling.
    '''
    def generateNegSamples(self, repProba=0.5, exProba=0.5, repSeed=0, exSeed=0, headSeed=0, tailSeed=0):
        assert repProba >= 0 and repProba <= 1.0 and exProba >= 0 and exProba <= 1.0
        # Generate negtive samples from positive samples
        print("INFO : Generate negtive samples from positive samples.")
        self.negDf = self.posDf.copy()
        np.random.seed(repSeed)
        repProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negDf), ))
        np.random.seed(exSeed)
        exProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negDf), ))
        shuffleHead = self.negDf["head"].sample(frac=1.0, random_state=headSeed)
        shuffleTail = self.negDf["tail"].sample(frac=1.0, random_state=tailSeed)

        # Replacing head or tail
        def replaceHead(relHead, shuffHead, shuffTail, repP, exP):
            if repP >= repProba:
                '''
                Not replacing head.self.negD
                '''
                return relHead
            else:
                if exP > exProba:
                    '''
                    Replacing head with shuffle head.
                    '''
                    return shuffHead
                else:
                    '''
                    Replacing head with shuffle tail.
                    '''
                    return shuffTail
        def replaceTail(relTail, shuffHead, shuffTail, repP, exP):
            if repP < repProba:
                '''
                Not replacing tail.
                '''
                return relTail
            else:
                if exP > exProba:
                    '''
                    Replacing tail with shuffle tail.
                    '''
                    return shuffTail
                else:
                    '''
                    Replacing head with shuffle head.
                    '''
                    return shuffHead

        self.negDf["head"] = list(map(replaceHead, self.negDf["head"], shuffleHead, shuffleTail, repProbaDistribution, exProbaDistribution))
        self.negDf["tail"] = list(map(replaceTail, self.negDf["tail"], shuffleHead, shuffleTail, repProbaDistribution, exProbaDistribution))

    '''
    Used to transform CSV data to index-form
    ==> csvData : Input CSV data
    ==> repDict : A dict like {column_name : dict(entity_dict)}.
                  The keys are names of the csv columns, the corresponding
                  value is entity/relation dictionary which used to transform
                  entity/realtion to index.
    '''
    @staticmethod
    def transformToIndex(csvData:pd.DataFrame, repDict:dict):
        for col in repDict.keys():
            csvData[col] = csvData[col].apply(lambda x:repDict[col][x])

    def __len__(self):
        return len(self.posDf)

    def __getitem__(self, item):
        if hasattr(self, "negDf"):
            return np.array(self.posDf.iloc[item,:3]), np.array(self.negDf.iloc[item,:3])
        else:
            return np.array(self.posDf.iloc[item,:3])