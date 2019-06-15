# -*- coding :utf-8 -*-

import os
import json
import torch
import codecs
import pickle
import argparse
import numpy as np
from config import Config
from code.utils import utils
from code.models import TransE, TransH, TransA, TransD, KG2E
from code.utils import evaluation
from code.dataloader.dataloader import tripleDataset
from torch.utils.data import DataLoader

from torch.autograd import Variable
from tensorboardX import SummaryWriter
'''
parser = argparse.ArgumentParser()
# Data arguments
parser.add_argument("-pp", "--pospath", help="Positive triples path", type=str, default="./data/freebase_mtr100_mte100-train.txt")
parser.add_argument("-vp", "--validpath", help="Evaluation triples path", type=str, default="./data/freebase_mtr100_mte100-valid.txt")
parser.add_argument("-ep", "--entpath", help="Entity dict path", type=str, default="./source/dict/entityDict.json")
parser.add_argument("-rp", "--relpath", help="Relation dict path", type=str, default="./source/dict/relationDict.json")
parser.add_argument("-st", "--savetype", help="Embedding save type", type=str, default="txt", choices=["txt"])
# Dataloader arguments
parser.add_argument("-bs", "--batchsize", help="Batch size", type=int, default=64)
parser.add_argument("-sf", "--shuffle", help="Shuffle data", action="store_true")
parser.add_argument("-nw", "--numworkers", help="Num subprocess to load data", type=int, default=0)
parser.add_argument("-dl", "--droplast", help="Drop last data", action="store_true")
parser.add_argument("-RP", "--repproba", help="Probability of replacing head", type=float, default=0.5)
parser.add_argument("-EP", "--exproba", help="Probability of exchanging head with tail", type=float, default=0.5)
# Model and training arguments
parser.add_argument("-ug", "--usegpu", help="Use GPU", action="store_true")
parser.add_argument("-gn", "--gpunum", help="GPU number", type=int, default=0)
parser.add_argument("-mn", "--modelname", help="Model name", type=str, default="TransE")
parser.add_argument("-ed", "--embeddingdim", help="Embedding dimension", type=int, default=128)
parser.add_argument("-wd", "--weightdecay", help="Weight decay", type=float, default=0.0)
parser.add_argument("-mg", "--margin", help="Margin value in margin-ranking loss", type=float, default=1.0)
parser.add_argument("-ds", "--distance", help="Adopt L1 or L2 to measure the distance", type=int, default=2, choices=[1,2])
parser.add_argument("-e",  "--epochs", help="Training epochs", type=int, default=15)
parser.add_argument("-ee", "--evalepoch", help="Evaluation epochs", type=int, default=1)
parser.add_argument("-lr", "--learningrate", help="Learning rate", type=float, default=0.01)
parser.add_argument("-op", "--optimizer", help="Training optimizer", type=str, default="Adam")
parser.add_argument("-em", "--evalmethod", help="Evaluation method", type=str, default="MR", choices=["MR", "Hit10", "MRR"])
parser.add_argument("-sm", "--simmeasure", help="Similarity measure in evaluation", type=str, default="dot", choices=["dot", "cos", "L2"])
parser.add_argument("-ms", "--modelsave", help="Model saving method", type=str, default="param", choices=["param", "full"])
parser.add_argument("-mp", "--modelpath", help="Model save path", type=str, default="./source/model/")
parser.add_argument("-lp", "--loadembed", help="Load pre-training embedding", action="store_true")
parser.add_argument("-ef", "--entityfile", help="Pre-training entity embedding file", type=str, default="./source/embed/entityEmbedding.txt")
parser.add_argument("-rf", "--relationfile", help="Pre-training relation embedding file", type=str, default="./source/embed/relationEmbedding.txt")
parser.add_argument("-pm", "--premodel", help="Pre-training model path", type=str, default="./source/model/TransE_ent128_rel128.param")
# Other arguments
parser.add_argument("-sd", "--summarydir", help="Summary dirname", type=str, default="./source/summary/")

USE_CONFIG = True
if USE_CONFIG:
    args = Config()
else:
    args = parser.parse_args()
'''
args = Config()
def prepareDataloader(args, repSeed, exSeed, headSeed, tailSeed):
    # Initialize dataset and dataloader
    # If print(dataset[:]), you can get the result like:
    #   (np.array(N, 3, dtype=int64), np.array(N, 3, dtype=int64))
    # The first array represents the positive triples, while
    #   the second array represents the negtive ones.
    #   N is the size of all data.
    dataset = tripleDataset(posDataPath=args.pospath,
                            entityDictPath=args.entpath,
                            relationDictPath=args.relpath)
    dataset.generateNegSamples(repProba=args.repproba,
                               exProba=args.exproba,
                               repSeed=repSeed,
                               exSeed=exSeed,
                               headSeed=headSeed,
                               tailSeed=tailSeed)
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=args.shuffle,
                            num_workers=args.numworkers,
                            drop_last=args.droplast)
    return dataloader

def prepareEvalDataloader(args):
    dataset = tripleDataset(posDataPath=args.validpath,
                            entityDictPath=args.entpath,
                            relationDictPath=args.relpath)
    dataloader = DataLoader(dataset,
                            batch_size=len(dataset),
                            shuffle=False,
                            drop_last=False)
    return dataloader

def adjust_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

class trainTriples():
    def __init__(self, args):
        self.args = args

    def prepareData(self):
        print("INFO : Prepare dataloader")
        # self.dataloader = prepareDataloader(self.args)
        self.evalloader = prepareEvalDataloader(self.args)
        self.entityDict = json.load(open(self.args.entpath, "r"))
        self.relationDict = json.load(open(self.args.relpath, "r"))

    def prepareModel(self):
        print("INFO : Init model %s"%self.args.modelname)
        if self.args.modelname == "TransE":
            self.model = TransE.TransE(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       embeddingDim=self.args.TransE["EmbeddingDim"],
                                       margin=self.args.TransE["Margin"],
                                       L=self.args.TransE["L"])
        elif self.args.modelname == "TransH":
            self.model = TransH.TransH(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       embeddingDim=self.args.TransH["EmbeddingDim"],
                                       margin=self.args.TransH["Margin"],
                                       L=self.args.TransH["L"],
                                       C=self.args.TransH["C"],
                                       eps=self.args.TransH["Eps"])
        elif self.args.modelname == "TransA":
            self.model = TransA.TransA(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       embeddingDim=self.args.TransA["EmbeddingDim"],
                                       margin=self.args.TransA["Margin"],
                                       L=self.args.TransA["L"],
                                       lamb=self.args.TransA["Lamb"],
                                       C=self.args.TransA["C"])
        elif self.args.modelname == "TransD":
            self.model = TransD.TransD(entityNum=len(self.entityDict["stoi"]),
                                       relationNum=len(self.relationDict["stoi"]),
                                       entityDim=self.args.TransD["EntityDim"],
                                       relationDim=self.args.TransD["RelationDim"],
                                       margin=self.args.TransD["Margin"],
                                       L=self.args.TransD["L"])
        elif self.args.modelname == "KG2E":
            self.model = KG2E.KG2E(entityNum=len(self.entityDict["stoi"]),
                                   relationNum=len(self.relationDict["stoi"]),
                                   embeddingDim=self.args.KG2E["EmbeddingDim"],
                                   margin=self.args.KG2E["Margin"],
                                   sim=self.args.KG2E["Sim"],
                                   vmin=self.args.KG2E["Vmin"],
                                   vmax=self.args.KG2E["Vmax"])
        else:
            print("ERROR : No model named %s"%self.args.modelname)
            exit(1)
        if self.args.usegpu:
            with torch.cuda.device(self.args.gpunum):
                self.model.cuda()

    def loadPretrainEmbedding(self):
        if self.args.modelname == "TransE":
            print("INFO : Loading pre-training entity and relation embedding!")
            self.model.initialWeight(entityEmbedFile=self.args.entityfile,
                                     entityDict=self.entityDict["stoi"],
                                     relationEmbedFile=self.args.relationfile,
                                     relationDict=self.relationDict["stoi"])
        else:
            print("ERROR : Model %s is not supported!"%self.args.modelname)
            exit(1)

    # [TODO]Different models should be considered differently
    def loadPretrainModel(self):
        if self.args.modelname == "TransE":
            print("INFO : Loading pre-training model.")
            modelType = os.path.splitext(self.args.premodel)[-1]
            if modelType == ".param":
                self.model.load_state_dict(torch.load(self.args.premodel))
            elif modelType == ".model":
                self.model = torch.load(self.args.premodel)
            else:
                print("ERROR : Model type %s is not supported!")
                exit(1)
        else:
            print("ERROR : Model %s is not supported!" % self.args.modelname)
            exit(1)

    def fit(self):
        EPOCHS = self.args.epochs
        LR = self.args.learningrate
        OPTIMIZER = self.args.optimizer
        if OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         weight_decay=self.args.weightdecay,
                                         lr=LR)
        else:
            print("ERROR : Optimizer %s is not supported."%OPTIMIZER)
            exit(1)

        # Training, GLOBALSTEP and GLOBALEPOCH are used for summary
        minLoss = float("inf")
        bestMR = float("inf")
        GLOBALSTEP = 0
        GLOBALEPOCH = 0
        for seed in range(100):
            print("INFO : Using seed %d" % seed)
            self.dataloader = prepareDataloader(self.args, repSeed=seed, exSeed=seed, headSeed=seed, tailSeed=seed)
            for epoch in range(EPOCHS):
                GLOBALEPOCH += 1
                STEP = 0
                print("="*20+"EPOCHS(%d/%d)"%(epoch+1, EPOCHS)+"="*20)
                for posX, negX in self.dataloader:
                    # Allocate tensor to devices
                    if self.args.usegpu:
                        with torch.cuda.device(self.args.gpunum):
                            posX = Variable(torch.LongTensor(posX).cuda())
                            negX = Variable(torch.LongTensor(negX).cuda())
                    else:
                        posX = Variable(torch.LongTensor(posX))
                        negX = Variable(torch.LongTensor(negX))

                    # Normalize the embedding if neccessary
                    self.model.normalizeEmbedding()

                    # Calculate the loss from the model
                    loss = self.model(posX, negX)
                    if self.args.usegpu:
                        lossVal = loss.cpu().item()
                    else:
                        lossVal = loss.item()

                    # Calculate the gradient and step down
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Print infomation and add to summary
                    if minLoss > lossVal:
                        minLoss = lossVal
                    print("[TRAIN-EPOCH(%d/%d)-STEP(%d)]Loss:%.4f, minLoss:%.4f"%(epoch+1, EPOCHS, STEP, lossVal, minLoss))
                    STEP += 1
                    GLOBALSTEP += 1
                    sumWriter.add_scalar('train/loss', lossVal, global_step=GLOBALSTEP)
                if GLOBALEPOCH % self.args.lrdecayepoch == 0:
                    adjust_learning_rate(optimizer, decay=self.args.lrdecay)
                if GLOBALEPOCH % self.args.evalepoch == 0:
                    MR = evaluation.MREvaluation(evalloader=self.evalloader,
                                                 model=self.args.modelname,
                                                 simMeasure=args.simmeasure,
                                                 **self.model.retEvalWeights())
                    sumWriter.add_scalar('train/eval', MR, global_step=GLOBALEPOCH)
                    print("[EVALUATION-EPOCH(%d/%d)]Measure method %s, eval %.4f"% \
                          (epoch+1, EPOCHS, self.args.evalmethod, MR))
                    # Save the model if new MR is better
                    if MR < bestMR:
                        bestMR = MR
                        self.saveModel()
                        self.dumpEmbedding()

    def saveModel(self):
        if self.args.modelsave == "param":
            path = os.path.join(self.args.modelpath, "{}_ent{}_rel{}.param".format(self.args.modelname, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"]))
            torch.save(self.model.state_dict(), path)
        elif self.args.modelsave == "full":
            path = os.path.join(self.args.modelpath, "{}_ent{}_rel{}.model".format(self.args.modelname, getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"]))
            torch.save(self.model, path)
        else:
            print("ERROR : Saving mode %s is not supported!"%self.args.modelsave)
            exit(1)

    def dumpEmbedding(self):
        '''
        TXT save type only supports saving embedding and relation embedding
        '''
        if self.args.savetype == "txt":
            entWeight = self.model.entityEmbedding.weight.detach().cpu().numpy()
            relWeight = self.model.relationEmbedding.weight.detach().cpu().numpy()
            entityNum, entityDim = entWeight.shape
            relationNum, relationDim = relWeight.shape
            entsave = os.path.join(self.args.embedpath, "entityEmbedding.txt")
            relsave = os.path.join(self.args.embedpath, "relationEmbedding.txt")
            with codecs.open(entsave, "w", encoding="utf-8") as fp:
                fp.write("{} {}\n".format(entityNum, entityDim))
                for ent, embed in zip(self.entityDict["itos"], entWeight):
                    fp.write("{}\t{}\n".format(ent, ",".join(embed.astype(np.str))))
            with codecs.open(relsave, "w", encoding="utf-8") as fp:
                fp.write("{} {}\n".format(relationNum, relationDim))
                for rel, embed in zip(self.relationDict["itos"], relWeight):
                    fp.write("{}\t{}\n".format(rel, ",".join(embed.astype(np.str))))
        elif self.args.savetype == "pkl":
            '''
            pkl saving type dump a dict containing itos list and weights returned by model
            '''
            pklPath = os.path.join(self.args.embedpath, "param_ent{}_rel{}_{}.pkl".format(getattr(self.args, self.args.modelname)["EmbeddingDim"], getattr(self.args, self.args.modelname)["EmbeddingDim"], self.model))
            with codecs.open(pklPath, "wb") as fp:
                pickle.dump({"entlist" : self.entityDict["itos"],
                             "rellist" : self.relationDict["itos"],
                             "weights" : self.model.retEvalWeights()}, fp)
        else:
            print("ERROR : Format %s is not supported."%self.args.savetype)
            exit(1)

if __name__ == "__main__":
    # Print args
    utils.printArgs(args)

    sumWriter = SummaryWriter(log_dir=args.summarydir)
    trainModel = trainTriples(args)
    trainModel.prepareData()
    trainModel.prepareModel()
    if args.loadembed:
        trainModel.loadPretrainEmbedding()
    trainModel.fit()

    sumWriter.close()