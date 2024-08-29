import random
import numpy as np

def true_output(x):
    return (10*x+2)

def noisy_obs(x_lo, x_hi, Np, sigma):
    x_list=[]
    t_list=[]
    for i in range (0,Np):
        x = random.uniform(x_lo,x_hi)
        t = true_output(x)+random.gauss(0,sigma)
        x_list.append(x)
        t_list.append(t)
    return(np.array(x_list), np.array(t_list))

def poly_model(a,b,c,d,e,f,x):
    return(a+b*x+c*x**2+d*x**3+e*x**4+f*x**5)

def ModelMatrix(Np,obs_x):
    N = np.ones((Np,6))
    for i in range (0,Np):
        N[i,1] = obs_x[i]
        N[i,2] = obs_x[i]**2
        N[i,3] = obs_x[i]**3
        N[i,4] = obs_x[i]**4
        N[i,5] = obs_x[i]**5
    return (N)

class Basis():
    def Linear(Np,x):
        temp = np.ones((Np,2))
        for i in range(0,Np):
            temp[i,1] = x[i]
        return(temp)
    def Quadratic(Np,x):
        temp = np.ones((Np,3))
        for i in range(0,Np):
            temp[i,1] = x[i]
            temp[i,2] = x[i]**2
        return(temp)
    def Cubic(Np,x):
        temp = np.ones((Np,4))
        for i in range(0,Np):
            temp[i,1] = x[i]
            temp[i,2] = x[i]**2
            temp[i,3] = x[i]**3
        return(temp)
    def Forth(Np,x):
        temp = np.ones((Np,5))
        for i in range(0,Np):
            temp[i,1] = x[i]
            temp[i,2] = x[i]**2
            temp[i,3] = x[i]**3
            temp[i,4] = x[i]**4
        return(temp)
    def Fifth(Np,x):
        temp = np.ones((Np,6))
        for i in range(0,Np):
            temp[i,1] = x[i]
            temp[i,2] = x[i]**2
            temp[i,3] = x[i]**3
            temp[i,4] = x[i]**4
            temp[i,5] = x[i]**4
        return(temp)

def BayLinReg(Np,obs_x):
    N = np.ones((Np,2))
    for i in range (0,Np):
        N[i,1] = obs_x[i]
    return (N)

class poly():
    def Linear(x,w0,w1):
        return(w0+w1*x)
    def Quadratic (x,w0,w1,w2):
        return(w0+w1*x+w2*x*x)
    def Cubic (x,w0,w1,w2,w3):
        return(w0+w1*x+w2*x*x+w3*x*x*x)
    def Forth (x,w0,w1,w2,w3,w4):
        return(w0+w1*x+w2*x**2+w3*x**3+w4*x**4)
    def Fifth (x,w0,w1,w2,w3,w4,w5):
        return(w0+w1*x+w2*x**2+w3*x**3+w4*x**4+w5*x**5)

def E(t,Phi,w,beta,alpha):
    temp1 = t-np.matmul(Phi,w)
    temp2 = 0.5*beta*np.matmul(temp1.T,temp1)
    temp3 = 0.5*alpha*np.matmul(w.T,w)
    return(temp2+temp3)

def Evidence(t,Phi,w,beta,alpha,M,N,A):
    term3 = E(t,Phi,w,beta,alpha)
    term1 = 0.5*M*np.log(alpha)
    term2 = 0.5*N*np.log(beta)
    term4 = 0.5*np.log(np.linalg.det(A))
    term5 = 0.5*N*np.log(2*np.pi)
    tmp = term1+term2-term3-term4-term5
    return(tmp)