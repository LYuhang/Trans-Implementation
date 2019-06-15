# -*- coding: utf-8 -*-

import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, entityNum, relationNum, embeddingDim, margin=1.0, L=2):
        super(TransE, self).__init__()
        assert (L == 1 or L == 2)
        self.model = "TransE"
        self.margin = margin
        self.L = L

        self.entityEmbedding = nn.Embedding(num_embeddings=entityNum,
                                            embedding_dim=embeddingDim)
        self.relationEmbedding = nn.Embedding(num_embeddings=relationNum,
                                              embedding_dim=embeddingDim)
        self.distfn = nn.PairwiseDistance(L)

    '''
    This function used to calculate score, steps follows:
    ==> Step1: Split input as head, relation and tail index column
    ==> Step2: Transform index tensor to embedding tensor
    ==> Step3: Sum head, relation and tail tensors with weights (1, 1, -1)
    ==> Step4: Calculate distance as final score
    '''
    def scoreOp(self, inputTriple):
        # Step1
        # head : shape(batch_size, 1)
        # relation : shape(batch_size, 1)
        # tail : shape(batch_size, 1)
        head, relation, tail = torch.chunk(input=inputTriple,
                                           chunks=3,
                                           dim=1)
        # Step2
        # head : shape(batch_size, 1, embedDim)
        # relation : shape(batch_size, 1, embedDim)
        # tail : shape(batch_size, 1, embedDim)
        head = torch.squeeze(self.entityEmbedding(head), dim=1)
        tail = torch.squeeze(self.entityEmbedding(tail), dim=1)
        relation = torch.squeeze(self.relationEmbedding(relation), dim=1)

        # Step3 and Step4
        # output : shape(batch_size, embedDim) ==> shape(batch_size, 1)
        output = self.distfn(head+relation, tail)
        return output

    '''
    In every training epoch, the entity embedding should be normalize
    first. There are three steps:
    ==> Step1: Get numpy.array from embedding weight
    ==> Step2: Normalize array
    ==> Step3: Assign normalized array to embedding
    '''
    def normalizeEmbedding(self):
        embedWeight = self.entityEmbedding.weight.detach().cpu().numpy()
        embedWeight = embedWeight / np.sqrt(np.sum(np.square(embedWeight), axis=1, keepdims=True))
        self.entityEmbedding.weight.data.copy_(torch.from_numpy(embedWeight))

    def retEvalWeights(self):
        return {"entityEmbed": self.entityEmbedding.weight.detach().cpu().numpy(),
                "relationEmbed": self.relationEmbedding.weight.detach().cpu().numpy()}

    '''
    Input description:
    ==> posX : (torch.tensor)The positive triples tensor, shape(batch_size, 3)
    ==> negX : (torch.tensor)The negtive triples tensor, shape(batch_size, 3)
    '''
    def forward(self, posX, negX):
        size = posX.size()[0]
        # Calculate score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        # Get margin ranking loss
        # max(posScore-negScore+margin, 0)
        return torch.sum(F.relu(input=posScore-negScore+self.margin)) / size

    '''
    Used to load pretraining entity and relation embedding.
    Implementation steps list as following:
    Method one: (Assign the pre-training vector one by one)
    ==> Step1: Read one line at a time, split the line as entity string and embed vector.
    ==> Step2: Transform the embed vector to np.array
    ==> Step3: Look up entityDict, find the index of the entity from entityDict, assign 
               the embed vector from step1 to the embedding matrix
    ==> Step4: Repeat steps above until all line are checked.
    Method two: (Assign the pre-training at one time)
    ==> Step1: Initial a weight with the same shape of the embedding matrix
    ==> Step2: Read every line of the EmbedFile and assign the vector to the intialized 
               weight.
    ==> Step3: Assign the intialized weight to the embedding matrix at one time after
               all line are checked.
    '''
    def initialWeight(self, entityEmbedFile, entityDict, relationEmbedFile, relationDict, fileType="txt"):
        print("INFO : Loading entity pre-training embedding.")
        with codecs.open(entityEmbedFile, "r", encoding="utf-8") as fp:
            _, embDim = fp.readline().strip().split()
            assert int(embDim) == self.entityEmbedding.weight.size()[-1]
            for line in fp:
                ent, embed = line.strip().split("\t")
                embed = np.array(embed.split(","), dtype=float)
                if ent in entityDict:
                    self.entityEmbedding.weight.data[entityDict[ent]].copy_(torch.from_numpy(embed))
        print("INFO : Loading relation pre-training embedding.")
        with codecs.open(relationEmbedFile, "r", encoding="utf-8") as fp:
            _, embDim = fp.readline().strip().split()
            assert int(embDim) == self.relationEmbedding.weight.size()[-1]
            for line in fp:
                rel, embed = line.strip().split("\t")
                embed = np.array(embed.split(","), dtype=float)
                if rel in entityDict:
                    self.relationEmbedding.weight.data[relationDict[rel]].copy_(torch.from_numpy(embed))
