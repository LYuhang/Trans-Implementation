# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransH(nn.Module):
    def __init__(self, entityNum, relationNum, embeddingDim, margin=1.0, L=2, C=1.0, eps=0.001):
        super(TransH, self).__init__()
        assert (L in [1, 2])
        self.model = "TransH"
        self.entnum = entityNum
        self.relnum = relationNum
        self.margin = margin
        self.L = L
        self.C = C
        self.eps = eps

        self.entityEmbedding = nn.Embedding(num_embeddings=entityNum,
                                            embedding_dim=embeddingDim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relationNum,
                                              embedding_dim=embeddingDim)
        self.relationHyper = nn.Embedding(num_embeddings=relationNum,
                                          embedding_dim=embeddingDim)
        self.distfn = nn.PairwiseDistance(L)

    '''
    Calculate the score:
    Step1 : Split the triple as head, relation and tail
    Step2 : Transform index tensor to embedding tensor
    Step3 : Project entity head and tail embedding to relation hyperplane
    Step4 : Calculate similarity score in relation hyperplane
    Step5 : Return the score
    '''
    def scoreOp(self, inputTriple):
        # Step1
        head, relation, tail = torch.chunk(inputTriple,
                                           chunks=3,
                                           dim=1)
        # Step2
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        relHyper = torch.squeeze(self.relationHyper(relation), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)
        # Step3
        head = head - relHyper * torch.sum(head * relHyper, dim=1, keepdim=True)
        tail = tail - relHyper * torch.sum(tail * relHyper, dim=1, keepdim=True)
        # Step4
        return self.distfn(head+relation, tail)

    '''
    Normalize relation hyper-plane embedding
    '''
    def normalizeEmbedding(self):
        hyperWeight = self.relationHyper.weight.detach().cpu().numpy()
        hyperWeight = hyperWeight / np.sqrt(np.sum(np.square(hyperWeight), axis=1, keepdims=True))
        self.relationHyper.weight.data.copy_(torch.from_numpy(hyperWeight))

    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy(),
                "hyperEmbed": self.relationHyper.weight.detach().cpu().numpy()}

    def forward(self, posX, negX):
        size = posX.size()[0]
        # Calculate score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        # Get margin ranking loss
        # max(posScore-negScore+margin, 0)
        # Use F.relu()
        marginLoss = torch.sum(F.relu(input=posScore-negScore+self.margin))
        entityLoss = torch.sum(F.relu(torch.norm(self.entityEmbedding.weight, p=2, dim=1, keepdim=False)-1))
        orthLoss = torch.sum(F.relu(torch.sum(self.relationHyper.weight * self.relationEmbedding.weight, dim=1, keepdim=False) / \
                                    torch.norm(self.relationEmbedding.weight, p=2, dim=1, keepdim=False) - self.eps ** 2))
        return  marginLoss/size +self.C*(entityLoss/self.entnum + orthLoss/self.relnum)
