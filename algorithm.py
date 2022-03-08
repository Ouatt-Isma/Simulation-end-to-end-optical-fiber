import math 
import itertools
from parameters import parameters
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft
import scipy as sc 
# from scipy.fftpack import fft #or from scipy.fftpack import fft
# from scipy import fft 
# from scipy import ifft 
from numpy.fft import fftshift,fft, ifft
import sys


# def __init__(self, _L= 1000*(10**3), _B = 20, _adb= 0.2, 
#                  _D = 17*(10**3), _gama= 1.27*(10**(-3)), 
#                  _lambda0=1.55*(10**(-6)), _P = 6*(10**(-3)),
#                  _T=400, _N=2**10):
       
            
def power(const):
    return np.sum(np.linalg.norm(const, axis=1)**2)/len(const)
## L = 1000km or 1 km, calcul adb, log(10)=1 ? 
def qam(P, M1=4,M2=4): #return list of point of the constellation
    constellation1 = []
    constellation2 = []
    #verify that M1 and M2 are power of two
    assert(math.ceil(np.log2(M1)) == math.floor(np.log2(M1))) 
    for k in range(M1):
        constellation1.append((2*k+1-M1))
    for k in range(M2):
        constellation2.append((2*k+1-M2))
    constellation = np.array(list(itertools.product(constellation1,constellation2)))
    constellation=(np.sqrt(P)/np.sqrt(power(constellation)))*constellation
    #assert power(constellation) == P
    return constellation
def source(N, p):
    return np.random.binomial(size=N, n=1, p=1-p)

def modulator(data, M):
    # Constants
    sqrt_M = np.sqrt(M).astype(int)
    k = np.log2(M).astype(int)
    vect = np.array(range(sqrt_M))
    gray_constallation = np.bitwise_xor(vect, np.floor(vect/2).astype(int))

    vect = np.arange(1, np.sqrt(M), 2)
    symbols = np.concatenate((np.flip(-vect, axis=0), vect)).astype(int)

    data_input = data.reshape((-1,k))
    I = np.zeros((data_input.shape[0],))
    Q = np.zeros((data_input.shape[0],))
    for n in range(int(data_input.shape[1] / 2)):
        I = I + data_input[:,n] * 2 ** n
    for n in range(int(data_input.shape[1]/2),int(data_input.shape[1])):
        Q = Q + data_input[:,n] * 2 ** (n - int(data_input.shape[1]/2))
    I = I.astype(int)
    Q = Q.astype(int)
    I = gray_constallation[I]
    Q = gray_constallation[Q]
    I = symbols[I]
    Q = symbols[Q]
    S = (I + 1j * Q)    
    return S
def gray_mapper_qam(P, M = 16, verbose=False):
    if(M==2):
        return {'0': [-P, 0], '1': [P, 0]}
    k = np.log2(M).astype(int)
    a = {}
    for j in range(M):
        bits = bin(j)[2:]
        bits = '0'*(k - len(bits)) + bits 
        t =  modulator(np.array([int(i) for i in bits]), M)
        a[bits] =[t.real[0], t.imag[0]]
    tmp_a = np.array(list(a.values()))
    amp = np.sqrt(P)/np.sqrt(power(tmp_a))
    for j in range(M):
        bits = bin(j)[2:]
        bits = '0'*(k - len(bits)) + bits 
        a[bits] =[a[bits][0]*amp, a[bits][1]*amp]
    if (verbose):
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data',0))
        b = list(a.values())
        ax.scatter([b[i][0] for i in range(len(b))], [b[i][1] for i in range(len(b))])
        for i in a:
            ax.annotate(i, a[i])
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.annotate('$\mathcal{R}(s)$', xy=(1, 0.5), ha='left', va='top', xycoords='axes fraction', fontsize=20)
        ax.annotate('$\mathcal{I}(s)$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
        plt.title("Gray Mapping for "+str(M)+"-QAM constellation")
        plt.show()
    return a
def bit_to_symb(b, cnt): 
    n = len(b)
    k = len(list(cnt.keys())[0])
    assert (n%k == 0)
    block_bits = [b[i:i+k] for i in range(0, n, k)]
    res = []
    for i in block_bits:
        res.append(cnt[ ''.join([str(bt) for bt in i]) ])
    return np.array(res)

# def mod(params, s):
#     Ns = len(s) # number of symbols
#     l1 = -math.floor(Ns/2)
#     l2 = math.ceil(Ns/2)-1
#     q0t = np.zeros(len(params.t), dtype=complex)
#     i = 0
#     for ti in params.t:
#         q0ti = 0
#         for l in range(l1, l2+1):
#             tmp= s[l-l1][0] + 1j*s[l-l1][1]
#             q0ti += tmp*np.sinc(params.B*params.T0*ti-l)
#         q0t[i] = q0ti
#         i+=1
#     return q0t

    
# def mod(params, s):
#     Ns = len(s) # number of symbols
#     l1 = -math.floor(Ns/2)
#     l2 = math.ceil(Ns/2)-1
#     q0t = np.zeros(len(params.t), dtype=complex)
#     #for ti in params.t:
#     #q0t = np.zeros(len(params.t))
#     for l in range(l1, l2+1):
#         tmp= s[l-l1][0] + 1j*s[l-l1][1]
#         q0t += tmp*np.sinc(params.Bn*params.t-l)
#     return q0t

def mod(params, s):
    Ns = len(s) # number of symbols
    l1 = -math.floor(Ns/2)
    l2 = math.ceil(Ns/2)-1
    q0t = np.zeros(len(params.t), dtype=complex)
    i = 0
    for ti in params.t:
        q0ti = 0
        for l in range(l1, l2+1):
            tmp= s[l-l1][0] + 1j*s[l-l1][1]
            q0ti += tmp*np.sinc(params.Bn*ti-l)

        q0t[i] = q0ti
        i+=1
    
    return q0t

def channel(params, q0t):
    z = params.L
    a = params.sigma2*params.Bn*z # total noise power in B Hz and distance [0, z]
    #f = np.linspace(-1/(2*params.dt), 1/(2*params.dt), len(params.t))  # get the f vector from t vector
    q0f = fft(q0t)#fft.fft(q0t)
    h = np.exp(1j*(params.f**2)*z)
    qzf = q0f*h # output in frequency
    # add Guassian noise in frequency, with correct variance
    N0 = a/2
    #print(N0)
    std = np.sqrt(N0)
    qzf = qzf + np.random.normal(0, std, len(params.t)) + 1j*np.random.normal(0, std, len(params.t))
    qzt = ifft(qzf)#fft.ifft(qzf) # back to time
    return qzt, qzf

def compare(a, b):
    assert len(a) == len(b)
    return (np.linalg.norm(a-b))/len(a)

def equalize(params, qzt):
    z = params.L
    ##f = np.linspace(-1/(2*params.dt), 1/(2*params.dt), len(params.t)) # get the f vector from t vector
    qzf = fft(qzt) #fft.fft(qzt) # input in frequency
    h_minus1 = np.exp(-1j*(params.f**2)*z)
    qzfe = qzf*h_minus1 # output in frequency
    qzte = ifft(qzfe) #fft.ifft(qzfe)  # back to time
    return qzte, qzfe 

# def demod(params, qzte, Ns):
#     shat = np.zeros(Ns, dtype=complex)
#     l1 = -math.floor(Ns/2)
#     l2 = math.ceil(Ns/2)-1
#     i = 0
#     for l in range(l1, l2+1):
#         shat[i] = np.sum(qzte*np.sinc(params.B*params.T0*params.t - l)*params.dt)
#         i+=1
#     return shat 

def demod(params, qzte, Ns):
    shat = np.zeros(Ns, dtype=complex)
    l1 = -math.floor(Ns/2)
    l2 = math.ceil(Ns/2)-1
    i = 0
    for l in range(l1, l2+1):
        shat[i] = params.Bn*np.sum(qzte*np.sinc(params.Bn*params.t - l)*params.dt)
        i+=1
    return shat 

def tostring_cp(b):
    return str(b.real)+" "+str(b.imag)+'j'

def thres(i, cnt):
    Lcnt = list(set(np.array(list(cnt.values()))[:,0]))
    dmin = np.abs(Lcnt[0] - i)
    res = Lcnt[0]
    for cpt in range(1, len(Lcnt)):
        if(np.abs(Lcnt[cpt] - i) < dmin): 
            dmin = np.abs(Lcnt[cpt] - i)
            res= Lcnt[cpt]
    return res
def estimate(si, cnt):
    Lcnt = list(set(np.array(list(cnt.values()))[:,0]))
    p = np.abs(Lcnt[0])
    if(len(cnt)==2):
        if(si.real>0):
            return [p, 0]
        else:
            return [-p, 0]
    return [thres(si.real, cnt), thres(si.imag, cnt)]
def detector(shat, cnt):
    Ns =len(shat)
    stilde = np.zeros((Ns,2))
    for i in range(Ns):
        stilde[i] = estimate(shat[i], cnt)
        
    return stilde  

def symb_to_bit(stilde, cnt):
    symbol_map_inv={}
    for i in cnt:
        symbol_map_inv[tostring_cp(cnt[i][0] + 1j* cnt[i][1])] = i
    Ns = len(stilde)
    bhat = []
    for i in range(Ns):
        bhat.append(symbol_map_inv[tostring_cp(stilde[i][0] + 1j*stilde[i][1])])
    return np.array([int(i) for i in ''.join(bhat)])

def ber(b, bhat):
    assert len(b) == len(bhat)
    tmp_size = len(b)
    res=0
    for i in range(tmp_size):
        if(b[i] != bhat[i]):
            res+=1
    return res/tmp_size
def ser(s, shat): 
    assert len(s) == len(shat)
    tmp_size = len(s)
    res=0
    for i in range(tmp_size):
        if(s[i] != shat[i]):
            res+=1
    return res/tmp_size

def Ber(params, nber_sample=10, ns=10000, p=0.5, M=2):
    res = 0
    ns=10#0#0#0
    nb = np.log2(M).astype(int)*ns
    cnt = gray_mapper_qam(params.P, M)
    for i in range(nber_sample):
        b= source(nb, p)
        s = bit_to_symb(b, cnt)
        q0t = mod(params, s)
        qzt,_ = channel(params, q0t)
        qzte,_ = equalize(params, qzt); # equalized output
        shat = demod(params, qzte, ns)
        stilde = detector(shat, cnt)
        bhat = symb_to_bit(stilde, cnt)
        #print(ber_one(b, bhat))
        res+=ber(b, bhat)
    return res/nber_sample, (shat, s)

##### Question 20 #####
def noise(params, dz):
    var = (params.sigma2/2)*dz*params.Bn
    std= np.sqrt(var)
    return np.random.normal(0, std, len(params.t)) + 1j*np.random.normal(0, std, len(params.t))

##### Question 21 #####
def sigma(x, eps):
    return x*np.exp(-2j*eps*(np.abs(x)**2))

def sigma_minus1(y, eps):
    r = np.abs(y)
    phi = np.angle(y) + 2*eps*(r**2)
    return r*np.exp(1j*phi)
    
def nnet_gen(x, params, nz=1000, z=1): 
    n = len(x)
    w = fftshift(2*np.pi*params.f) # angular frequency vector
    #w = 2*np.pi*params.f
    dz = z/nz # epsilon
    h = np.exp(1j*(w**2)*dz) #all-pass filter
    D = dft(n)/np.sqrt(n) # DFT matrix of size n
    # W = (np.linalg.inv(D))@np.diag(h)@D
    W = D.transpose().conjugate()@np.diag(h)@D
    #Loop over the nnet layers
    for k in range(nz):
        # linear transformation -- multiplication by the weight matrix W
        x = W@x
        # activation function
        x = sigma(x, dz)
        # noise addition
        x = x + noise(params, dz)
    return x

def nnet_pred(x, params, nz=1): 
    n = len(x)
    w = 2*np.pi*params.f # angular frequency vector
    z = 1
    dz = z/nz # epsilon
    h = np.exp(1j*(w**2)*dz) #all-pass filter
    D = dft(n)/np.sqrt(n) # DFT matrix of size n
    # print(np.shape(np.diag(h)))
    # print(np.shape(D.transpose().conjugate()))
    # D.transpose().conjugate()@np.diag(h)
    # np.diag(h)@D
    W = D.transpose().conjugate()@np.diag(h)@D
    
    W_minus = np.linalg.inv(W)
    #Loop over the nnet layers
    for k in range(nz):
        # linear transformation -- multiplication by the weight matrix W
        x = sigma_minus1(x, dz)
        x = W_minus@x
        # activation function
        
        # noise addition
    
    
    return x