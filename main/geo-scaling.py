from func.genn import get_real_A_with_real_alpha, plot_p_vs_distance
import numpy as np
import matplotlib.pyplot as plt


def main(num, ave_deg, alpha=0, lambda_normed=0):

    A, cluster, _ = get_real_A_with_real_alpha(num=num,alpha=alpha,lambda_normed=lambda_normed,k=ave_deg,sd=2,parallel_edges=False,method='exact',f=1)
    plot_p_vs_distance(A, cluster,alpha, n_bin=50,label_size=17)
    plt.tight_layout()
    plt.show()

main(num=1000, ave_deg=20, alpha=0.7)