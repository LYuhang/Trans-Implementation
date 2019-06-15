# -*- coding: utf-8 -*-

import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TransD(nn.Module):
    def __init__(self, entityNum, relationNum, entityDim, relationDim, margin=1.0, L=2):
        super(TransD, self).__init__()
        self.model = "TransD"
        self.margin = margin
        self.L = L

        # Initialize the entity and relation embedding and projection embedding
        self.entityEmbedding = nn.Embedding(num_embeddings=entityNum,
                                            embedding_dim=entityDim)
        self.entityMapEmbedding = nn.Embedding(num_embeddings=entityNum,
                                             embedding_dim=entityDim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relationNum,
                                              embedding_dim=relationDim)
        self.relationMapEmbedding = nn.Embedding(num_embeddings=relationNum,
                                                 embedding_dim=relationDim)

        self.distfn = nn.PairwiseDistance(L)

    '''
    Calculate the score, steps are follows:
    Step1: Split input triples as head, relation and tail
    Step2: Calculate the mapping matrix Mrh and Mrt
    Step3: Calculate the mapping vector of head and tail
    Step4: Return the score
    '''
    def scoreOp(self, inputTriples):
        head, relation, tail = torch.chunk(inputTriples,
                                           chunks=3,
                                           dim=1)
        headp = torch.squeeze(self.entityMapEmbedding(head), dim=1)   # (B, 1, En) -> (B, En)
        head = torch.squeeze(self.entityEmbedding(head), dim=1)       # (B, 1, En) -> (B, En)
        tailp = torch.squeeze(self.entityMapEmbedding(tail), dim=1)   # (B, 1, En) -> (B, En)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)       # (B, 1, En)  -> (B, En)
        relationp = torch.squeeze(self.relationMapEmbedding(relation), dim=1) # (B, 1, Em) -> (B, Em)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)     # (B, 1, Em) -> (B, Em)

        '''
        relationp(B, Em) -> relationp(B, Em, 1)
        headp(B, En) -> headp(B, 1, En)
        tailp(B, En) -> tailp(B, 1, En)
        (B, Em, 1) * (B, 1, En) -> (B, Em, En)
        '''
        relatioDim = relation.size()[1]
        entityDim = head.size()[1]
        relationp = torch.unsqueeze(relationp, dim=2)   # (B, Em, 1)
        headp = torch.unsqueeze(headp, dim=1)           # (B, 1, En)
        tailp = torch.unsqueeze(tailp, dim=1)           # (B, 1, En)
        if inputTriples.is_cuda:
            Mrh = torch.matmul(relationp, headp) + torch.eye(relatioDim, entityDim).cuda(inputTriples.device.index)
            Mrt = torch.matmul(relationp, tailp) + torch.eye(relatioDim, entityDim).cuda(inputTriples.device.index)
        else:
            Mrh = torch.matmul(relationp, headp) + torch.eye(relatioDim, entityDim)
            Mrt = torch.matmul(relationp, tailp) + torch.eye(relatioDim, entityDim)
        # Map head and tail with mapping Matrix Mrh and Mrt
        # Mrh, Mrt : (B, Em, En)
        # head, tail : (B, En) -> (B, En, 1)
        # (B, Em, En) * (B, En, 1) -> (B, Em, 1)
        head = torch.unsqueeze(head, dim=2)
        tail = torch.unsqueeze(tail, dim=2)
        head = torch.squeeze(torch.matmul(Mrh, head), dim=2)   # (B, Em, 1) -> (B, Em)
        tail = torch.squeeze(torch.matmul(Mrt, tail), dim=2)   # (B, Em, 1) -> (B, Em)
        output = self.distfn(head+relation, tail)
        return output

    '''
    Normalize embedding
    '''
    def normalizeEmbedding(self):
        self.entityEmbedding.weight.data.copy_(torch.renorm(input=self.entityEmbedding.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1))
        self.relationEmbedding.weight.data.copy_(torch.renorm(input=self.relationEmbedding.weight.detach().cpu(),
                                                              p=2,
                                                              dim=0,
                                                              maxnorm=1))

    '''
    Return evaluation weights
    '''
    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "entityMapEmbed": self.entityMapEmbedding.weight.detach().cpu().numpy(),
                "relationMapEmbed": self.relationMapEmbedding.weight.detach().cpu().numpy()}

    def forward(self, posX, negX):
        size = posX.size()[0]

        # Calculate score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        return torch.sum(F.relu(input=posScore-negScore+self.margin)) / size


