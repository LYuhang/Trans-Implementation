# -*- coding: utf-8 -*-

import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class KG2E(nn.Module):
    def __init__(self, entityNum, relationNum, embeddingDim, margin=1.0, sim="KL", vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        assert (sim in ["KL", "EL"])
        self.model = "KG2E"
        self.margin = margin
        self.sim = sim
        self.ke = embeddingDim
        self.vmin = vmin
        self.vmax = vmax

        # Embeddings represent the mean vector of entity and relation
        # Covars represent the covariance vector of entity and relation
        self.entityEmbedding = nn.Embedding(num_embeddings=entityNum,
                                            embedding_dim=embeddingDim)
        self.entityCovar = nn.Embedding(num_embeddings=entityNum,
                                        embedding_dim=embeddingDim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relationNum,
                                              embedding_dim=embeddingDim)
        self.relationCovar = nn.Embedding(num_embeddings=relationNum,
                                          embedding_dim=embeddingDim)

    '''
    Calculate the KL loss between T-H distribution and R distribution.
    There are four parts in loss function.
    '''
    def KLScore(self, **kwargs):
        # Calculate KL(e, r)
        losep1 = torch.sum(kwargs["errorv"]/kwargs["relationv"], dim=1)
        losep2 = torch.sum((kwargs["relationm"]-kwargs["errorm"])**2 / kwargs["relationv"], dim=1)
        KLer = (losep1 + losep2 - self.ke) / 2

        # Calculate KL(r, e)
        losep1 = torch.sum(kwargs["relationv"]/kwargs["errorv"], dim=1)
        losep2 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / kwargs["errorv"], dim=1)
        KLre = (losep1 + losep2 - self.ke) / 2
        return (KLer + KLre) / 2

    '''
    Calculate the EL loss between T-H distribution and R distribution.
    There are three parts in loss function.
    '''
    def ELScore(self, **kwargs):
        losep1 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / (kwargs["errorv"] + kwargs["relationv"]), dim=1)
        losep2 = torch.sum(torch.log(kwargs["errorv"]+kwargs["relationv"]), dim=1)
        return (losep1 + losep2) / 2

    '''
    Calculate the score of triples
    Step1: Split input as head, relation and tail index
    Step2: Transform index tensor to embedding
    Step3: Calculate the score with "KL" or "EL"
    Step4: Return the score 
    '''
    def scoreOp(self, inputTriples):
        head, relation, tail = torch.chunk(input=inputTriples,
                                           chunks=3,
                                           dim=1)

        headm = torch.squeeze(self.entityEmbedding(head), dim=1)
        headv = torch.squeeze(self.entityCovar(head), dim=1)
        tailm = torch.squeeze(self.entityEmbedding(tail), dim=1)
        tailv = torch.squeeze(self.entityCovar(tail), dim=1)
        relationm = torch.squeeze(self.relationEmbedding(relation), dim=1)
        relationv = torch.squeeze(self.relationCovar(relation), dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        if self.sim == "KL":
            return self.KLScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        elif self.sim == "EL":
            return self.ELScore(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        else:
            print("ERROR : Sim %s is not supported!" % self.sim)
            exit(1)

    def normalizeEmbedding(self):
        self.entityEmbedding.weight.data.copy_(torch.renorm(input=self.entityEmbedding.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        self.relationEmbedding.weight.data.copy_(torch.renorm(input=self.relationEmbedding.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1.0))
        self.entityCovar.weight.data.copy_(torch.clamp(input=self.entityCovar.weight.detach().cpu(),
                                                       min=self.vmin,
                                                       max=self.vmax))
        self.relationCovar.weight.data.copy_(torch.clamp(input=self.relationCovar.weight.detach().cpu(),
                                                         min=self.vmin,
                                                         max=self.vmax))

    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "entityCovar": self.entityCovar.weight.detach().cpu().numpy(),
                "relationCovar": self.relationCovar.weight.detach().cpu().numpy(),
                "Sim":self.sim}

    def forward(self, posX, negX):
        size = posX.size()[0]

        # Calculate score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        return torch.sum(F.relu(input=posScore-negScore+self.margin)) / size
