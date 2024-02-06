import numpy as np
import matplotlib.pyplot as plt
from func.genn import get_real_A_with_real_alpha
from func.entropy import info_entropy, get_energy_bound
import scipy

def main_line(num, ave_deg, type='send'):

    alpha_list = np.linspace(0,2,20)
    ent = []
    s_td =[]

    for alpha_idx in range(len(alpha_list)):
        alpha = alpha_list[alpha_idx]       
        s_e = []
        for sd in range(1000):
            lambda_normed=0
            keep_f = 1
            A, cluster, _ = get_real_A_with_real_alpha(num=num,alpha=alpha,lambda_normed=lambda_normed,k=ave_deg,sd=2,parallel_edges=False,method='exact',f=keep_f,seed=sd)
            d = scipy.spatial.distance_matrix(cluster, cluster)
            d = d.reshape(-1)
            w = np.mean(d)*ave_deg
            new_A = get_energy_bound(A,cluster,w=w)
            if type == 'send':
                entropy = info_entropy(new_A)
            elif type == 'receive':
                entropy = info_entropy(new_A.T)
            s_e.append(np.mean(entropy))

        ent.append(np.mean(s_e))
        s_td.append(np.std(s_e))
        
    x=alpha_list
    y=ent
    plt.errorbar(x, y, yerr=s_td, fmt='o:', markersize=8, capsize=5, elinewidth=1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r'$\alpha$',fontsize=20)
    plt.ylabel(r'$Entropy$',fontsize=20)
    plt.show()


num = 1000
ave_deg = 20
type = 'receive' # 'send' send or receive
main_line(num, ave_deg, type=type)
