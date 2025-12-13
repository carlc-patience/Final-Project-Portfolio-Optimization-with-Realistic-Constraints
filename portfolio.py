import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from estimator import *
import cvxpy as cp
import heapq

class UnionFind:
    def __init__(self, nodes):
        self.parents = {node : node for node in nodes}
        self.sizes = {node : 1 for node in nodes}
    
    def find(self, node):
        if self.parents[node] == node:
            return node
        parent = self.parents[node]
        self.parents[node] = self.find(parent)
        return self.parents[node]
    
    def union(self, node1, node2):
        root1, root2 = self.find(node1), self.find(node2)

        if root1 == root2: return
        
        size1, size2 = self.sizes[root1], self.sizes[root2]
        if size1 < size2:
            self.parents[root1] = root2
            self.sizes[root2] = size1 + size2
        else:
            self.parents[root2] = root1
            self.sizes[root1] = size1 + size2
    
    def getSize(self, node):
        return self.sizes[self.find(node)]


class portfolioConstructor:
    def __init__(self, longOnly=True, useCustomSLinkage=False):
        self.longOnly = longOnly
        self.useCustomSLinkage = useCustomSLinkage

    def fit(self, X, estimator='sample', optimizer="minVar"):
        """
        general function to calculate the weights of the portfolio uisng Min-Variance Method / HRP
        """
        assert X.ndim == 2, "X must be 2D of shape (N, T)"
        if estimator == 'sample':
            covMatrix = sampleCovEstimator.computeCovMatrix(X)
        elif estimator == 'shrinkage':
            covMatrix = shrinkageCovEstimator.computeCovMatrix(X)
        else:
            raise Exception('Covariance Not Supported')

        if optimizer == "minVar":
            weights = self.calcMinVarWeights(covMatrix)
        elif optimizer == "hrp":
            weights = self.calcHrpWeights(covMatrix)
        else:
            raise Exception("Optimizer Not Supported")

        return weights
    
    def calcHrpWeights(self, covMatrix):
        """
        This is the main procedure for HRP method, containing 3 steps
        """
        link = self.treeClustering(covMatrix)
        sortedIndex = self.quasi_diagonalization(link)
        hrpWeights = self.recurBisection(covMatrix, sortedIndex)
        return np.array(hrpWeights)
    

    def calcMinVarWeights(self, covMatrix):
        """
        calculates the closed-form global minimum-variance portfolio weights
        """
        cov = np.asarray(covMatrix, dtype=float)
        cov = 0.5 * (cov + cov.T)
        N = cov.shape[0]
        ones = np.ones(N)
        covInv = np.linalg.pinv(cov)
        weight = covInv @ ones
        denom = ones @ weight
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            weight = np.ones(N) / N
        else:
            weight = weight / denom

        if self.longOnly:
            weight= np.clip(weight, 0, None)
            s = weight.sum()
            if s <= 1e-12:
                weight[:] = 1.0 / N
            else:
                weight = weight / s
        return weight



    def treeClustering(self, covMatrix):
        """
        Tree Clustering Step
        citations: chatgpt indicated using the ssd.squareform, sch.linkage package, I handcrafted a version myself(much slower though, SEE self.SingleLinkage Function)
        """
        std = np.sqrt(np.diag(covMatrix))
        corrMatrix = covMatrix / np.outer(std, std)
        corrMatrix = np.clip(covMatrix, -1, 1)

        dist1 = np.sqrt(0.5 * (1 - corrMatrix))
        distHat = dist1.copy()
        if self.useCustomSLinkage:
            link = self.singleLinkage(distHat)
        else: 
            dist_condensed = ssd.squareform(distHat, checks=False)
            link = sch.linkage(dist_condensed, method="single")
        return link


    def singleLinkage(self, distMatrix):
        """
        Single linkage using UnionFind
        """
        assert distMatrix.ndim == 2, "The distance matrix must be 2D"

        N = distMatrix.shape[0]
        nodes = list(range(N))
        NodeReprMap = {newNode: (newNode if newNode < N else None) for newNode in range(2 * N - 1)}

        uf = UnionFind(nodes)
        clusterMembers = {i: {i} for i in range(N)}
        activeNodes = set(range(N))
        linkageMatrix = np.zeros((N - 1, 4))

        K = N
        mergeStep = 0
        dists = []
        rowIndices, colIndices = np.triu_indices(N, k=1)
        for row, col in zip(rowIndices, colIndices):
            distance = distMatrix[row, col]
            heapq.heappush(dists, (distance, row, col))

        while len(activeNodes) > 1 and mergeStep < N - 1 and dists:
            dist, node1, node2 = heapq.heappop(dists)
            if node1 not in activeNodes or node2 not in activeNodes or node1 == node2:
                continue
            repr1 = NodeReprMap[node1]
            repr2 = NodeReprMap[node2]
            root1 = uf.find(repr1)
            root2 = uf.find(repr2)
            if root1 == root2:
                continue

            uf.union(root1, root2)
            newRoot = uf.find(root1)
            members1 = clusterMembers.pop(root1)
            members2 = clusterMembers.pop(root2)
            newMembers = members1 | members2
            clusterMembers[newRoot] = newMembers
            NodeReprMap[K] = newRoot
            activeNodes.remove(node1)
            activeNodes.remove(node2)
            activeNodes.add(K)
            clusterSize = len(newMembers)
            linkageMatrix[mergeStep] = [node1, node2, dist, clusterSize] if node1 < node2 else [node2, node1, dist, clusterSize]
            mergeStep += 1

            # update the distance between clusters
            for otherNode in list(activeNodes):
                if otherNode == K:
                    continue
                otherRoot = uf.find(NodeReprMap[otherNode])
                otherMembers = clusterMembers[otherRoot]
                minDist = np.inf
                for member in newMembers:
                    rowDists = distMatrix[member, list(otherMembers)]
                    curMin = np.min(rowDists)
                    if curMin < minDist:
                        minDist = curMin
                heapq.heappush(dists, (minDist, K, otherNode))
            K += 1

        return linkageMatrix
    

    def quasi_diagonalization(self, link):
        """
        Quasi-Diagonalization Function
        """
        linkageMatrix = link.astype(int)
        numAssets = linkageMatrix[-1, 3]           
        rootNode = int(2 * numAssets - 2)  
        nodeQueue = [rootNode]                  
        sortedIndex = []                          

        while nodeQueue:
            node = nodeQueue.pop(0)
            if node < numAssets:
                sortedIndex.append(node)
            else:
                leftChild = int(linkageMatrix[node - numAssets, 0])
                rightChild = int(linkageMatrix[node - numAssets, 1])
                nodeQueue.extend([leftChild, rightChild])

        return sortedIndex


    def recurBisection(self, covMatrix, sortedIndex):
        """
        Recursive Bisection method to generate the weights
        citation: I gave the instructions and general idea, used chatgpt to generate relevant function
        """
        sortedCov = covMatrix[np.ix_(sortedIndex, sortedIndex)] 
        N = sortedCov.shape[0]

        def calcClusterVar(indices):
            sub_cov = sortedCov[np.ix_(indices, indices)]
            inv_var = 1.0 / np.diag(sub_cov)
            w = inv_var / inv_var.sum()
            return w.T @ sub_cov @ w

        def allocateWeight(start, end):
            if end - start == 1:
                return np.array([1.0])
            mid = (start + end) // 2
            left_idx = list(range(start, mid))
            right_idx = list(range(mid, end))
            w_left = allocateWeight(start, mid)
            w_right = allocateWeight(mid, end)
            var_left = calcClusterVar(left_idx)
            var_right = calcClusterVar(right_idx)
            alpha_left = 1.0 - var_left / (var_left + var_right)
            alpha_right = 1.0 - alpha_left
            w_left *= alpha_left
            w_right *= alpha_right

            return np.concatenate([w_left, w_right])

        w_sorted = allocateWeight(0, N)
        w_sorted /= w_sorted.sum()

        w = np.zeros(N)
        w[sortedIndex] = w_sorted
        return w

