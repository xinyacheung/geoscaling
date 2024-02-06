import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import os 
env = os.path.dirname(__file__)

def f_1(x, A, B):
    return A * x + B

def plot_p_vs_distance(A, cluster,alpha, n_bin=50,label_size=17):
    
    d_i_j_mat = scipy.spatial.distance_matrix(cluster, cluster)
    d0 = d_i_j_mat[d_i_j_mat > 0]
    d1 = d_i_j_mat[A > 0]

    dis_1 = list(np.log10(d1))
    dis_0 = list(np.log10(d0))

    mx = max(max(dis_1),max(dis_0))
    mn = min(min(dis_1),min(dis_0))
    print(mx) # the maximum distance
    print(mn) # the minimum distance
    bins = np.linspace(mn, mx, n_bin)

    dis_1 = [i for i in dis_1 if i>=mn and i<=mx]
    dis_0 = [i for i in dis_0 if i>=mn and i<=mx]

    [dis_1_num,binn] = np.histogram(dis_1,bins)
    [dis_0_num,binn] = np.histogram(dis_0,bins)
    p_conn=[]
    for i in range(0,len(dis_1_num)):
        if dis_0_num[i] != 0:
            p_conn.append(dis_1_num[i]/dis_0_num[i])
        else:
            p_conn.append(0)
    b = list(binn)
    bb=[]
    p=[]
    for i in range(0,len(b)-1):
        if p_conn[i]!=0:
            p.append(np.log10(p_conn[i]))
            bb.append((b[i+1]-b[i])/2+b[i])

    col = 'k' 
    bb=[10**i for i in bb]
    p =[10**i for i in p]
    plt.figure()
    plt.plot(bb,p,'o',color=col,alpha=0.7, markerfacecolor='none')
    plt.yscale('log')
    plt.xscale('log')

    plt.title(r'$\alpha$='+str(alpha),fontsize=label_size)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.xlabel('Distance',fontsize=label_size)
    plt.ylabel('Connected probability',fontsize=label_size)

def power_weight(A):
    alpha = 3.0  # power-law index
    x_min = 1  # minimum value
    random_numbers = np.random.pareto(alpha, np.sum(A==1)) + x_min
    k=0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i!=j and A[i,j] ==1:
                A[i,j] = random_numbers[k]
                k+=1
    return A

def get_real_A_with_real_alpha(num,alpha,lambda_normed,k=20,sd=1,name='whole',parallel_edges=False,method='exact',f=1,seed=123):
    
    path = env + '/dataset/' ## load neuron locations
    points = np.load(path+f'{name}.npy')
    points = points.astype(float)
    num = num 
    np.random.seed(seed)
    selected_rows = np.random.choice(points.shape[0], num, replace=False)
    cluster0 = points[selected_rows, :]
    
    max_possible_dist = np.sqrt(np.sum((np.max(points, axis=0) - np.min(points, axis=0))**2))
    norm_const = max_possible_dist
    
    cluster = cluster0/max_possible_dist
    
    k = int(np.round(k/f))
    
    N = cluster.shape[0]
    num_links = int(np.round(np.random.normal(N*k, np.sqrt(N)*sd)))

    q_j_mat = []
    r_i_list = []
    r_i_j_mat = []
    for i in range(N):
        cluster_i = cluster[i]
        disp_array_i = cluster - cluster_i
        dist_array_i = np.sqrt(np.sum(disp_array_i**2, axis=1))
        dist_array_i[i] = np.inf
        r_i_j_list = dist_array_i**(-alpha)*np.exp(-lambda_normed*dist_array_i)
        r_i_j_list[i] = 0
        r_i_j_mat = r_i_j_mat + [r_i_j_list]
        r_i = np.sum(r_i_j_list)
        r_i_list = r_i_list + [r_i]
        q_j_list = list(r_i_j_list/r_i)
        q_j_mat = q_j_mat + [q_j_list]
    p_i_list = np.array(r_i_list)/np.sum(r_i_list)
    p_i_list = p_i_list/np.sum(p_i_list)
    

    if parallel_edges:
        print('Considering multigraph')
        replace = True
    else:
        print('Considering simple graph')
        replace = False
    
    if (method == 'approx'):
        source_node_list = np.random.choice(N, num_links, replace=True, p=p_i_list)
        outdeg_list, _ = np.histogram(source_node_list, np.arange(-0.5, N + 1, 1))
        while (not parallel_edges) and any(outdeg_list > (N - 1)):
            source_node_list = np.random.choice(N, num_links, replace=True, p=p_i_list)
            outdeg_list, _ = np.histogram(source_node_list, np.arange(-0.5, N + 1, 1))
        A = np.zeros((N, N))
        for i in range(N):
            outdeg = outdeg_list[i]
            q_j_list = q_j_mat[i]
            j_list = np.random.choice(N, outdeg, replace=replace, p=q_j_list)
            j_list = np.random.choice(j_list, int(np.round(f*outdeg)), replace=False)
            [A_i, _] = np.histogram(j_list, np.arange(-0.5, N + 0.5))
            A[i] = A_i
    elif (method == 'exact'):
        r_i_j_mat = np.array(r_i_j_mat)
        p_i_j_mat = r_i_j_mat/np.sum(r_i_j_mat)
        p_i_j_mat = p_i_j_mat/np.sum(p_i_j_mat)
        p_i_j_list = np.reshape(p_i_j_mat, N**2)
        nonzero_A_i_j_list = np.random.choice(N**2, num_links, replace=replace, p=p_i_j_list)
        nonzero_A_i_j_list = np.random.choice(nonzero_A_i_j_list, int(np.round(f*num_links)), replace=False)
        A = np.zeros(N**2)
        [A, _] = np.histogram(nonzero_A_i_j_list, np.arange(-0.5, N**2 + 0.5))
        A = np.reshape(A, (N, N))
     
    A = power_weight(A) # get weights from power-law
    
    lam = lambda_normed/max_possible_dist

    return A, cluster0, lam