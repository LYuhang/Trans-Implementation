# -*- coding: utf-8 -*-

import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TransA(nn.Module):
    def __init__(self, entityNum, relationNum, embeddingDim, margin=1.0, L=2, lamb=0.01, C=0.2):
        super(TransA, self).__init__()
        assert (L==1 or L==2)
        self.model = "TransE"
        self.entnum = entityNum
        self.relnum = relationNum
        self.enbdim = embeddingDim
        self.margin = margin
        self.L = L
        self.lamb = lamb
        self.C = C

        self.entityEmbedding = nn.Embedding(num_embeddings=entityNum,
                                            embedding_dim=embeddingDim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relationNum,
                                              embedding_dim=embeddingDim)
        self.distfn = nn.PairwiseDistance(L)

    '''
    Normalize embedding
    '''
    def normalizeEmbedding(self):
        pass

    '''
    Reset Wr to zero
    '''
    def resetWr(self, usegpu, index):
        if usegpu:
            self.Wr = torch.zeros((self.relnum, self.enbdim, self.enbdim)).cuda(index)
        else:
            self.Wr = torch.zeros((self.relnum, self.enbdim, self.enbdim))

    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "Wr": self.Wr.detach().cpu().numpy()}

    '''
    Calculate the Mahalanobis distance weights
    '''
    def calculateWr(self, posX, negX):
        size = posX.size()[0]
        posHead, posRel, posTail = torch.chunk(input=posX,
                                               chunks=3,
                                               dim=1)
        negHead, negRel, negTail = torch.chunk(input=negX,
                                               chunks=3,
                                               dim=1)
        posHeadM, posRelM, posTailM = self.entityEmbedding(posHead), \
                                   self.relationEmbedding(posRel), \
                                   self.entityEmbedding(posTail)
        negHeadM, negRelM, negTailM = self.entityEmbedding(negHead), \
                                   self.relationEmbedding(negRel), \
                                   self.entityEmbedding(negTail)
        errorPos = torch.abs(posHeadM + posRelM - posTailM)
        errorNeg = torch.abs(negHeadM + negRelM - negTailM)
        del posHeadM, posRelM, posTailM, negHeadM, negRelM, negTailM
        self.Wr[posRel] += torch.sum(torch.matmul(errorNeg.permute((0, 2, 1)), errorNeg), dim=0) - \
                           torch.sum(torch.matmul(errorPos.permute((0, 2, 1)), errorPos), dim=0)
        '''
        # (B, 1, E) -> (B, E, 1) * (B, 1, E) -> (B, E, E)
        errorPos = torch.abs(posHead+posRel-posTail)
        errorNeg = torch.abs(negHead+negRel-negTail)
        del posHead, posRel, posTail, negHead, negRel, negTail
        Wr = torch.sum(torch.matmul(errorNeg.permute((0, 2, 1)), errorNeg), dim=0) / size - torch.sum(torch.matmul(errorPos.permute((0, 2, 1)), errorPos), dim=0) / size
        Wr = F.relu(input=Wr)
        return Wr
        '''

    '''
    This function is used to calculate score, steps follows:
    Step1: Split input as head, relation and tail index array
    Step2: Transform index array to embedding vector
    Step3: Calculate Mahalanobis distance weights
    Step4: Calculate distance as final score
    '''
    def scoreOp(self, inputTriples):
        head, relation, tail = torch.chunk(input=inputTriples,
                                           chunks=3,
                                           dim=1)
        relWr = self.Wr[relation]
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)

        # (B, E) -> (B, 1, E) * (B, E, E) * (B, E, 1) -> (B, 1, 1) -> (B, )
        error = torch.unsqueeze(torch.abs(head+relation-tail), dim=1)
        error = torch.matmul(torch.matmul(error, torch.unsqueeze(relWr, dim=0)), error.permute((0, 2, 1)))
        return torch.squeeze(error)

    def forward(self, posX, negX):
        size = posX.size()[0]
        self.calculateWr(posX, negX)

        # Calculate score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        # Calculate loss
        marginLoss = 1 / size * torch.sum(F.relu(input=posScore-negScore+self.margin))
        WrLoss = 1 / size * torch.norm(input=self.Wr, p=self.L)
        weightLoss = ( 1 / self.entnum * torch.norm(input=self.entityEmbedding.weight, p=2) + \
                       1 / self.relnum * torch.norm(input=self.relationEmbedding.weight, p=2))
        return marginLoss + self.lamb * WrLoss + self.C * weightLoss