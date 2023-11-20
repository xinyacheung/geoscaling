### constrcut adjacency matrix with p = d^\alpha

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from func.genn import get_real_A_with_real_alpha
from func.HRdyn import func

def main(num, ave_deg):
    MP=[]
    real_name=[]
    alpha_list = np.linspace(0,1.5,15)
    for alpha in alpha_list:
        for times in range(1): # average over times
            lambda_normed=0
            keep_f = 1         
            A, _, _ = get_real_A_with_real_alpha(num=num,alpha=alpha,lambda_normed=lambda_normed,k=ave_deg,sd=2,name='whole',parallel_edges=False,method='exact',f=keep_f)

            G = nx.from_numpy_array(A,create_using=nx.DiGraph())
            print(len(G.nodes))
            print(len(G.edges()))

            C = func( A )
            rows, cols = C.shape
            real_C = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    if i==j:
                        real_C[i,j] = 0
                    else:
                        real_C[i,j] = np.abs(C[i,j]) 

            ## participation coefficient
            communities_generator = nx.community.louvain_communities(G)
            cm = [list(s) for s in communities_generator]

            K = np.sum(real_C,axis=0)
            p=[]
            for i in list(G.nodes()):
                temp = 0
                for s in range(len(cm)):
                    Kis=0
                    for node_s in cm[s]:
                        Kis = real_C[i,node_s]+Kis
                    temp = temp + (Kis/K[i])**2
                p.append(1-temp)
            p = np.array(p)
            MP.append(np.mean(p))
            real_name.append(alpha)

    plt.figure()
    plt.plot(real_name,MP,'o')
    plt.ylabel('Mean participation coefficient')
    plt.xlabel(r'$\alpha$')
    plt.show()

main(num=1000, ave_deg=20)
