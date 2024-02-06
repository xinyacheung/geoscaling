import numpy as np
import networkx as nx

def get_energy_bound(A,cluster,w):
    n = A.shape[0] 
    D = np.zeros((n,n))
    new_A = np.zeros((n,n))
    for i in range(1,n-1):
        for j in range(i+1,n):
            D[i,j] = np.linalg.norm(cluster[i]-cluster[j])
            D[j,i] = D[i,j]
    
    for i in range(n):
        adj = A[i,:] 
        dis = D[i,:]
        
        sorted_indices = dis.argsort()[::-1] 
        sorted_dis = dis[sorted_indices]
        sorted_adj = adj[sorted_indices]
        
        energy_w = 0 # initial energy
        for l in range(n):
            if sorted_adj[l]!=0:
                new_A[i,l]=1
                energy_w = energy_w + sorted_dis[l]
                if energy_w >=w:
                    break
    
    return new_A

def info_entropy(A):
    np.fill_diagonal(A, 0) 
    nei = A
    nei_nei = A**2
    nei_node = nei + nei_nei
    row_sums = nei_node.sum(axis=1)

    row_sums[row_sums == 0] = 1 # avoid division by 0
    normalized_nei = nei_node / row_sums[:, np.newaxis]

    X= normalized_nei
    X[X == 0] = 1

    entropy = -np.sum(X * np.log(X), axis=1)

    return entropy
