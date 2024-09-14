import numpy as np
import matplotlib.pyplot as plt

def dist(x1,x2):
    t1 = x1-x2
    return(np.matmul(t1,t1.T))

def distComp(x):
    min_dist  = min(x)
    min_index = np.argmin(x)
    return(min_index, min_dist)


def plotCluster(X,rnk,N,mu_k,img_dir,iter):
    plt.figure(figsize=(6,6))
    plt.xlim(-12,12); plt.ylim(-12,12)
    plt.plot(mu_k[0][0],mu_k[0][1],'ro',mfc='w', markersize=14)
    plt.plot(mu_k[1][0],mu_k[1][1],'bo',mfc='w', markersize=14)
    plt.title('Cluster at '+str(iter),fontsize = 18) 
    for i in range(0,N):
        cluster  = np.argmax(rnk[i,:])
        if (cluster == 0):
            color = 'r'
        if (cluster == 1):
            color = 'b'
        plt.scatter(X[i,0], X[i,1], c = color,marker='x')
    plt.savefig(img_dir+'Cluster_at'+str(iter)+'.png')