
from algorithm import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  #pip install tqdm


def gen_data(n_sample=20000, nb = 64, M=16):
    print("-----------Data Generation----------- ")  
    p = 1/2
    ns=(nb/np.log2(M)).astype(int)
    cnt = gray_mapper_qam(params.P, M)
    Qzt = np.zeros((n_sample, params.N), dtype='complex')
    Q0t = np.zeros((n_sample, params.N), dtype='complex')
    S0t = np.zeros((n_sample, ns, 2))
    
    for i in tqdm(range(n_sample)):
        b= source(nb, p)
        s = bit_to_symb(b, cnt)
        q0t = mod(params, s)
        S0t[i] = s
        Q0t[i] = q0t    
        qzt = nnet_gen(q0t, params)
        Qzt[i] = qzt
    return Q0t, Qzt, S0t
if __name__=='__main__':
    params = parameters()
    # pp = parameters(_L= 1000000, _B = 10**6, _adb= 0.2, 
    #              _D = 17*(10**(3)), _gama= 1.27*(10**(-3)), 
    #              _lambda0=1.55*(10**(-6)), _P = 6*(10**(-3)),
    #              _T=400, _N=2**10)
    
    
    # params.Bn = pp.Bn
    # params.sigma2 = pp.sigma2
    # params.P = pp.P
    print("Data Generation with")
    print("sigma2=", params.sigma2)
    print("B=", params.B)
    print("Bn=", params.Bn)
    print("L=", params.L)
    print("P=", params.P)
    print("T0=", params.T0)
    print("SNR=", params.P/(params.sigma02*params.Bn*params.L))
    inp, outp, S = gen_data()
    # print(pp.sigma2)
    #tmp = nnet_pred(outp[0], params)
    #plt.plot(params.t, np.abs(inp[0]), label="Input") 
    # print(params.sigma2)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(params.t, np.abs(inp[0]), label="Input") 
    axs[1].plot(params.t, np.abs(outp[0]), label="Output") 
    
    # inp = np.exp(-params.t**2)
    # params.sigma2 = 0
    # inp = 1/np.cosh(params.t)
    # outp = nnet_gen(inp, params)
    # axs[0].plot(params.t, np.abs(inp), label="Input") 
    # axs[1].plot(params.t, np.abs(outp), label="Output") 
    
    np.save("input_signalvoila.npy", inp)
    np.save("input_symbolvoila.npy", S)
    np.save("output_signalvoila.npy", outp)
    
    plt.show()
    # plt.plot(params.t, np.abs(nnet_gen(inp[0], params)), label="Output") 
    # plt.show()
    
    # axs[1].plot(params.t, np.abs(tmp), label="Output") 
    # aa = np.concatenate((np.array(inp[0].real,ndmin=2).T,np.array(inp[0].imag,ndmin=2).T),axis=1)
    # print(aa[0])
    # print(inp[0][0])
    
    # plt.legend()
    
    
    
    # x = inp[0]
    # n = len(x)
    # eps = 10
    # y = sigma(x, eps)
    
    # xx= sigma_minus1(y, eps)
    # print(y)
    # print((x))
    # print((xx))
    
    # print(np.sum(np.abs(x - xx))/(n) )
    
    
    # nz = 1
    # x = inp[0]
    # n = len(x)
    # w = 2*np.pi*params.f # angular frequency vector
    # z = params.L
    # dz = z/nz # epsilon
    # h = np.exp(1j*(w**2)*dz) #all-pass filter
    # D = dft(n)/np.sqrt(n) # DFT matrix of size n
    # W = D.transpose().conjugate()@np.diag(h)@D
    # #Loop over the nnet layers
    # # linear transformation -- multiplication by the weight matrix W
    # x = W@x
    # # activation function
    # x = sigma(x, dz)
    # # noise addition
    # x = x + noise(params)
    
    # axs[0].plot(params.t, np.abs(x), label="Output") 
    # plt.show()
    
    
    
    
    

    
    # nz = 1
    # # dz = params.L
    # x = inp[0]
    # n = len(x)
    # w = 2*np.pi*params.f # angular frequency vector
    # z = params.L
    # dz = z/nz
    
    # h = np.exp(1j*(w**2)*dz) #all-pass filter
    # D = dft(n)/np.sqrt(n) # DFT matrix of size n
    # W = D.transpose().conjugate()@np.diag(h)@D
    # # #Loop over the nnet layers
    
    # # # linear transformation -- multiplication by the weight matrix W
    # # x1 = W@x
    # # # activation function
    # # x2 = sigma(x1, dz)
    # # # noise addition
    # # x2 = x2 + noise(params)
    
    # x2 = outp[0]
    
    # xx1 = sigma_minus1(x2, dz)
    
    
    # Wm1 = np.linalg.inv(W)
    # xx = Wm1@xx1
    # # print(np.sum(np.abs(x - xx))/(n) )
    # # fig, axs = plt.subplots(1, 2)
    # # axs[0].plot(params.t, np.abs(x), label="Input") 
    # axs[1].plot(params.t, np.abs(xx), label="Output") 
    # plt.show()