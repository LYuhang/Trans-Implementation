# -*- coding: utf-8 -*-

import torch
from code.utils.utils import CheckPath

class Config():
    def __init__(self):
        # Data arguments
        self.pospath = "./data/train.txt"
        self.validpath = "./data/valid.txt"
        self.entpath = "./source/dict/entityDict.json"
        self.relpath = "./source/dict/relationDict.json"
        self.embedpath = "./source/embed/"
        self.logpath = "./source/log/"
        self.savetype = "pkl"

        # Dataloader arguments
        self.batchsize = 1024
        self.shuffle = True
        self.numworkers = 0
        self.droplast = False
        self.repproba = 0.5
        self.exproba = 0.5

        # Model and training general arguments
        self.TransE = {"EmbeddingDim": 100,
                       "Margin":       1.0,
                       "L":            2}
        self.TransH = {"EmbeddingDim": 100,
                       "Margin":       1.0,
                       "L":            2,
                       "C":            0.01,
                       "Eps":          0.001}
        self.TransD = {"EntityDim":    100,
                       "RelationDim":  100,
                       "Margin":       2.0,
                       "L":            2}
        self.TransA = {"EmbeddingDim": 100,
                       "Margin":       3.2,
                       "L":            2,
                       "Lamb":         0.01,
                       "C":            0.2}
        self.KG2E   = {"EmbeddingDim": 100,
                       "Margin":       4.0,
                       "Sim":          "EL",
                       "Vmin":         0.03,
                       "Vmax":         3.0}
        self.usegpu = torch.cuda.is_available()
        self.gpunum = 0
        self.modelname = "KG2E"
        self.weightdecay = 0
        self.epochs = 5
        self.evalepoch = 1
        self.learningrate = 0.01
        self.lrdecay = 0.96
        self.lrdecayepoch = 5
        self.optimizer = "Adam"
        self.evalmethod = "MR"
        self.simmeasure = "L2"
        self.modelsave = "param"
        self.modelpath = "./source/model/"
        self.loadembed = False
        self.entityfile = "./source/embed/entityEmbedding.txt"
        self.relationfile = "./source/embed/relationEmbedding.txt"
        self.premodel = "./source/model/TransE_ent128_rel128.param"

        # Other arguments
        self.summarydir = "./source/summary/KG2E_EL/"

        # Check Path
        self.CheckPath()

        # self.usePaperConfig()

    def usePaperConfig(self):
        # Paper best params
        if self.modelname == "TransE":
            self.embeddingdim = 50
            self.learningrate = 0.01
            self.margin = 1.0
            self.distance = 1
            self.simmeasure = "L1"
        elif self.modelname == "TransH":
            self.batchsize = 1200
            self.embeddingdim = 50
            self.learningrate = 0.005
            self.margin = 0.5
            self.C = 0.015625
        elif self.modelname == "TransD":
            self.batchsize = 4800
            self.entitydim = 100
            self.relationdim = 100
            self.margin = 2.0

    def CheckPath(self):
        # Check files
        CheckPath(self.pospath)
        CheckPath(self.validpath)

        # Check dirs
        CheckPath(self.modelpath, raise_error=False)
        CheckPath(self.summarydir, raise_error=False)
        CheckPath(self.logpath, raise_error=False)
        CheckPath(self.embedpath, raise_error=False)


