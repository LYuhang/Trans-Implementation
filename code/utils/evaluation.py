# -*- coding: utf-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import dataloader

'''
Used to calculate the similarity between expected tail vector and real one.
==> expTailMatrix: shape(N, embedDim)Calculate by head and relation, 
                   N is the sample num(or batch size).
==> tailEmbedding: shape(entityNum, embedDim)The entity embedding matrix, 
                   entityNum is the number of entities.
==> return: shape(N, entityNum)The similarity between each vector in expTailMatrix 
            and all vectors in tailEmbedding.
'''
def calSimilarity(expTailMatrix:np.ndarray, tailEmbedding:np.ndarray, simMeasure="dot"):
    if simMeasure == "dot":
        return np.matmul(expTailMatrix, tailEmbedding.T)
    elif simMeasure == "cos":
        # First, normalize expTailMatrix and tailEmbedding
        # Then, use dot to calculate similarity
        return np.matmul(expTailMatrix / np.linalg.norm(expTailMatrix, ord=2, axis=1, keepdims=True),
                         (tailEmbedding / np.linalg.norm(expTailMatrix, ord=2, axis=1, keepdims=True)).T)
    elif simMeasure == "L2":
        simScore = []
        for expM in expTailMatrix:
            # expM :          (E, ) -> (1, E)
            # tailEmbedding : (N, E)
            score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=2, axis=1, keepdims=False)
            simScore.append(score)
        return np.array(simScore)
    elif simMeasure == "L1":
        simScore = []
        for expM in expTailMatrix:
            score = np.linalg.norm(expM[np.newaxis, :] - tailEmbedding, ord=1, axis=1, keepdims=False)
            simScore.append(score)
        return np.array(simScore)
    else:
        print("ERROR : Similarity method %s is not supported!"%simMeasure)
        exit(1)

'''
Used to calculate the rank of the right tail.
==> simScore: shape(N, entityNum)The similarity score.
==> tail: shape(N,)The real tail index.
==> simMeasure: Similarity method. Obviously, 
                the larger  the score, the higher/better the ranking when using "dot" or "cos",
                the smaller the score, the higher/better the ranking when using "L2".
There are three steps to implement:
==> Step1: Get the score of the real tails
==> Step2: Get the result of (simScore - realScore)
==> Step3: Count positive/negtive number of each line 
           as the rank of the real entity.
'''
def calRank(simScore:np.ndarray, tail:np.ndarray, simMeasure:str):
    realScore = simScore[np.arange(tail.shape[0]), tail].reshape((-1,1))
    judMatrix = simScore - realScore
    if simMeasure == "dot" or simMeasure == "cos":
        '''
        The larger the score, the better the rank.
        '''
        judMatrix[judMatrix > 0] = 1
        judMatrix[judMatrix < 0] = 0
        judMatrix = np.sum(judMatrix, axis=1)
        return judMatrix
    elif simMeasure == "L2" or simMeasure == "L1":
        '''
        The smaller the score, the better the rank
        '''
        judMatrix[judMatrix > 0] = 0
        judMatrix[judMatrix < 0] = 1
        judMatrix = np.sum(judMatrix, axis=1)
        return judMatrix
    else:
        print("ERROR : Similarity measure is not supported!")
        exit(1)

'''
Evaluation for TransE
'''
def evalTransE(head, relation, tail, simMeasure, **kwargs):
    # Use np.take() to gather embedding
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # tail = np.take(kwargs["entityEmbed"], indices=tail, axis=0)
    # Calculate the similarity, sort the score and get the rank
    simScore = calSimilarity(head+relation, kwargs["entityEmbed"], simMeasure=simMeasure)
    ranks = calRank(simScore, tail, simMeasure=simMeasure)
    return ranks

def calHyperSim(expTailMatrix, tailEmbedding, hyperMatrix, simMeasure="dot"):
    simScore = []
    for expM, hypM in zip(expTailMatrix, hyperMatrix):
        '''
        expM : shape(E,)
        hypM : shape(E,)
        Step1 : Projection tailEmbedding on hyperM as hyperTailEmbedding(shape(N,E))
        Step2 : Calculate similarity between expTailMatrix and hyperTailEmbedding
        Step3 : Add similarity to simMeasure
        (1, E) * matmul((N, E), (E, 1)) -> (1, E) * (N, 1) -> (N, E)
        '''
        hyperEmbedding = tailEmbedding - hypM[np.newaxis,:] * np.matmul(tailEmbedding, hypM[:,np.newaxis])
        if simMeasure == "dot":
            simScore.append(np.squeeze(np.matmul(hyperEmbedding, expM[:, np.newaxis])))
        elif simMeasure == "L2":
            # (E,) -> (1, E)
            # (N, E) - (1, E) -> (N, E)
            # np.linalg.norm()
            score = np.linalg.norm(hyperEmbedding-expM[np.newaxis, :], ord=2, axis=1, keepdims=False)
            simScore.append(score)
        else:
            print("ERROR : simMeasure %s is not supported!" % simMeasure)
            exit(1)
    return np.array(simScore)

'''
Evaluation for TransH
'''
def evalTransH(head, relation, tail, simMeasure, **kwargs):
    # Gather embedding
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    hyper = np.take(kwargs["hyperEmbed"], indices=relation, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Projection
    head = head - hyper * np.sum(hyper * head, axis=1, keepdims=True)
    simScore = calHyperSim(head+relation, kwargs["entityEmbed"], hyper, simMeasure=simMeasure)
    ranks = calRank(simScore, tail, simMeasure=simMeasure)
    return ranks

def calMapSim(expTailMatrix, tailEmbedding, tailMapMatrix, relMapMatrix, simMeasure="L2"):
    simScore = []
    for expM, relMap in tqdm(zip(expTailMatrix, relMapMatrix)):
        # relMap : (Em, ) -> (1, Em, 1)
        # tailMapMatrix : (N, En) -> (N, 1, En)
        # (1, Em, 1) * (N, 1, En) -> (N, Em, En)
        # (N, Em, En) * (N, En, 1) -> (N, Em, 1)
        # expM : (Em, )
        entdim = tailEmbedding.shape[1]
        reldim = relMapMatrix.shape[1]
        Mrt = np.matmul(relMap[np.newaxis, :, np.newaxis], tailMapMatrix[:, np.newaxis, :]) + np.eye(reldim, entdim)
        if simMeasure == "L2":
            score = np.linalg.norm(np.squeeze(np.matmul(Mrt, tailEmbedding[:,:,np.newaxis]), axis=2) - expM, ord=2, axis=1, keepdims=False)
            simScore.append(score)
        elif simMeasure == "L1":
            score = np.linalg.norm(np.squeeze(np.matmul(Mrt, tailEmbedding[:,:,np.newaxis]), axis=2) - expM, ord=1, axis=1, keepdims=False)
            simScore.append(score)
        else:
            print("ERROR : SimMeasure %s is not supported!" % simMeasure)
            exit(1)
    return np.array(simScore)

def evalTransD(head, relation, tail, simMeasure, **kwargs):
    # Gather embedding
    headp = np.take(kwargs["entityMapEmbed"], indices=head, axis=0)
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relationp = np.take(kwargs["relationMapEmbed"], indices=relation, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Generate Mapping Matrix
    # (B, Em) -> (B, Em, 1), (B, En) -> (B, 1, En)
    # (B, Em, 1) * (B, 1, En) -> (B, Em, En)
    # (B, En) -> (B, En, 1)
    # (B, Em, En) * (B, En, 1) -> (B, Em, 1)
    reldim = relation.shape[1]
    entdim = head.shape[1]
    Mrh = np.matmul(relationp[:, :, np.newaxis], headp[:, np.newaxis, :]) + np.eye(reldim, entdim)
    head = np.squeeze(np.matmul(Mrh, head[:, :, np.newaxis]), axis=2)
    simScore = calMapSim(head+relation, kwargs["entityEmbed"], kwargs["entityMapEmbed"], relationp, simMeasure=simMeasure)
    ranks = calRank(simScore, tail, simMeasure=simMeasure)
    return ranks

def calWeightSim(expTailMatrix, tailEmbedding, Wr):
    simScore = []
    for expM in tqdm(zip(expTailMatrix)):
        # expM : (E, )
        # (N, E) - (E, ) -> (N, E) -> abs() -> (N, E, 1), (N, 1, E)
        # (N, 1, E) * (N, E, E) * (N, E, 1) -> (N, 1, 1)
        error = np.abs(tailEmbedding - expM)
        score = np.squeeze(np.matmul(np.matmul(error[:,np.newaxis,:], Wr), error[:,:,np.newaxis]))
        simScore.append(score)
    return np.array(simScore)

def evalTransA(head, relation, tail, **kwargs):
    # Gather embedding
    head = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relation = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Calculate simScore
    simScore = calWeightSim(head+relation, kwargs["entityEmbed"], kwargs["Wr"])
    ranks = calRank(simScore, tail, simMeasure="L2")
    return ranks

def calKLSim(headMatrix, headCoMatrix, relationMatrix, relationCoMatrix, tailMatrix, tailCoMatrix, simMeasure="KL"):
    simScore = []
    for hM, hC, rM, rC in tqdm(zip(headMatrix, headCoMatrix, relationMatrix, relationCoMatrix)):
        # (N, E) - (E, )
        # (N, E) + (E, )
        errorm = tailMatrix - hM
        errorv = tailCoMatrix + hC
        if simMeasure == "KL":
            # (N, E) / (E, ) -> (N, E) -> sum() -> (N, )
            # (N, E) - (E, ) -> (N, E) ** 2 / (E, )
            score1 = np.sum(errorv / rC, axis=1, keepdims=False) + \
                    np.sum((rM - errorm)**2 / rC, axis=1, keepdims=False)
            score2 = np.sum(rC / errorv, axis=1, keepdims=False) + \
                    np.sum((rM - errorm)**2 / errorv, axis=1, keepdims=False)
            simScore.append((score1+score2)/2)
        elif simMeasure == "EL":
            score1 = np.sum((errorm-rM)**2 / (errorv+rC), axis=1, keepdims=False)
            score2 = np.sum(np.log(errorv+rC), axis=1, keepdims=False)
            simScore.append((score1+score2)/2)
    return np.array(simScore)

def evalKG2E(head, relation, tail, **kwargs):
    # Gather embedding
    headv = np.take(kwargs["entityCovar"], indices=head, axis=0)
    headm = np.take(kwargs["entityEmbed"], indices=head, axis=0)
    relationv = np.take(kwargs["relationCovar"], indices=relation, axis=0)
    relationm = np.take(kwargs["relationEmbed"], indices=relation, axis=0)
    # Calculate simScore
    simScore = calKLSim(headm, headv, relationm, relationv, kwargs["entityEmbed"], kwargs["entityCovar"], simMeasure=kwargs["Sim"])
    ranks = calRank(simScore, tail, simMeasure="L2")
    return ranks

'''
Implementation of MR metric, MR represents Mean Rank Metric
==> entityEmbed: Entity embedding matrix, shape(Ne, embedDim)
==> relationEmbed: Relation embedding matrix, shape(Nr, embedDim)
==> evalloader: Dataloader of evaluation triples
==> **kwargs : Neccessary model parameters used to evaluate
'''
def MREvaluation(evalloader:dataloader, model, simMeasure="dot", **kwargs):
    R = 0
    N = 0
    for tri in evalloader:
        # tri : shape(N, 3)
        # head : shape(N, 1) ==> shape(N)
        # relation : shape(N, 1) ==> shape(N)
        # tail : shape(N, 1) ==> shape(N)
        tri = tri.numpy()
        head, relation, tail = tri[:, 0], tri[:, 1], tri[:, 2]
        if model == "TransE":
            ranks = evalTransE(head, relation, tail, simMeasure, **kwargs)
        elif model == "TransH":
            ranks = evalTransH(head, relation, tail, simMeasure, **kwargs)
        elif model == "TransD":
            ranks = evalTransD(head, relation, tail, simMeasure, **kwargs)
        elif model == "TransA":
            ranks = evalTransA(head, relation, tail, **kwargs)
        elif model == "KG2E":
            ranks = evalKG2E(head, relation, tail, **kwargs)
        else:
            print("ERROR : The %s evaluation is not supported!" % model)
            exit(1)
        R += np.sum(ranks)
        N += ranks.shape[0]
    return (R / N)



