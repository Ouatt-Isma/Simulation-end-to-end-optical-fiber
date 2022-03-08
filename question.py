#from cProfile import label
## Env: Simul project
from cProfile import label
from re import I
from turtle import color
from algorithm import *
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import quad
#import sys
from collections import Counter
from scipy.fft import fft, ifft

from scipy.linalg import dft
#from numpy.fft import fftshift,fft, ifft

def report():
    print("Look at Report")
    
def see_alg():
    print("Look at algoritm.py")
    
def question1():
    print("-----------Question 1-----------")
    report()
    
def question2():
    print("-----------Question 2-----------")
    report()
    
def question3():
    print("-----------Question 3-----------")
    see_alg()

def question4(pp=6*(10**(-3))):
    print("-----------Question 4-----------")
    P = pp/params.P0
    print("normalize peak power is {:e}".format(P))
    a = qam(P)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    ax.scatter([a[i][0] for i in range(len(a))], [a[i][1] for i in range(len(a))])
    ax.set_xlim(-P, P)
    ax.set_ylim(-P, P)
    #plt.xlabel("Real part", loc='right')
    #ax.set_ylabel("Imaginary part", loc='top')
    plt.title("constellation C")
    ax.annotate('$\mathcal{R}(s)$', xy=(1, 0.5), ha='left', va='top', xycoords='axes fraction', fontsize=20)
    ax.annotate('$\mathcal{I}(s)$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
    ax.grid()
    plt.show()

def question5():
    print("-----------Question 5-----------: ")
    tmp = source(1024, 0.5)
    unique, counts = np.unique(tmp, return_counts=True)
    plt.grid()
    plt.hist(tmp)
    plt.title("Histogram of the source distribution for p=0.5")
    plt.xlabel(str(dict(zip(unique, counts))))
    plt.show()
    print(dict(zip(unique, counts)))
def question6():
    print("-----------Question 6-----------: ")
    b = source(1024, 0.5)
    cnt = gray_mapper_qam(params.P, verbose=True)
    #print(cnt)
    print(bit_to_symb(b, cnt))

def question7():
    print("-----------Question 7-----------")
    see_alg()
    # bits to signal
    M = 16 # size of the constellation
    n = 3 # number of symbols (or sinc functions); test with s=1
    nb = n*np.log2(M).astype(int) # number of bits
    p =1/2 # probability of zero
    b = source(nb, p) # Bernoulli source, random bits sequence
    cnt = gray_mapper_qam(params.P, M=M)
    #constellation
    s = bit_to_symb(b, cnt) # symbol sequence; could be inside modulator.m
    #print(s)
    q0t = mod(params, s) # transmitted signal
    q0f = fft(q0t)
    #q0f = dft(len(q0t))@q0t
    #print(q0f)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.title("Plot of q(t, 0) in time and in frequency", x=0.5, y=2.2)
    sample = range(int(params.N/2) - 50, int(params.N/2) + 50)
    ax1.plot(params.t[sample], np.abs(q0t[sample]))
    #ax1.plot(params.t, np.abs(q0t))
    ax2.plot(params.f, np.abs(q0f))
    ax1.set_xlabel("In time")
    ax2.set_xlabel("In frequency")
    
    plt.show()
    
def question8():
    print("-----------Question 8----------- ")
    report()
    
def question9():
    print("-----------Question 9----------- ")
    report()

def question10():
    print("-----------Question 10----------- ")
    report()

def question11():
    print("-----------Question 11----------- ")
    report()
    
def question12():
    print("-----------Question 12----------- ")
    see_alg()
  
def question13():
    print("-----------Question 13----------- ")
    report() 
 
def question14():
    #print("-----------Question 14----------- ") 
    #see_alg()
    A = 1
    q0t = A*np.exp(-params.t**2) # Gaussian input, for testing
    q0f = fft(q0t) # input in frequency
    # plot below the input in t & f. You must tune T and N!
    #ax = plt.gca()
    
    fig, (ax1, ax2) = plt.subplots(2, 2)
    

    ax1[0].plot(params.t, np.abs(q0t))
    ax1[0].set_xlabel("Input in Time")
    ax1[1].plot(params.f, np.abs(q0f))
    ax1[1].set_xlabel("Input in Frequency")
    sigma_save = params.sigma2
    params.sigma2 = 0
    params.L = 2000
    qzt, qzf = channel(params, q0t) # output in t,f. Zero noise.
    params.sigma2 = sigma_save 
    qzte, qzfe = equalize(params, qzt) # equalized output
    plt.title("Plot of the input and the equalized output", x=0, y=2.2)
    ax2[0].plot(params.t, np.abs(qzte)) # plot input & equalized output in t
    ax2[0].set_xlabel("Output equalized in Time")
    ax2[1].plot(params.f, np.abs(qzfe)) # plot input & equalized output in f
    ax2[1].set_xlabel("Output equalized in Frequency")
    print(compare(q0t, qzte))
    
    plt.show()

def question15():
    print("-----------Question 15----------- ")  
    #print(quad(lambda t: np.sinc(t)**2, -params.T/2, params.T/2)[0])
    # print(np.sum(qzte*np.sinc(params.B*params.t - l)*params.dt))
    print(np.sum((np.sinc(params.t)**2)*params.dt))
    print(np.sum((params.Bn*np.sinc(params.Bn*params.t)**2)*params.dt))
    
    
def question16():
    print("-----------Question 16----------- ")  
    
def question17():
    print("-----------Question 17----------- ")  
    
    # modulation
    nb = 64; # number of bits
    M = 16; # order of modulation
    ns = (nb/np.log2(M)).astype(int) # number of symbols
    p =1/2 # probability of zero
    b= source(nb, p)# random bit sequence
    # b = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
    #    1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
    #    0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1])
    cnt = gray_mapper_qam(params.P)
    #print(cnt)
    s = bit_to_symb(b, cnt)

    q0t = mod(params, s)
    #print(q0t[2])
    # propagation & equalization. Set the noise to zero for now
    params.sigma2=0
    qzt, qzf = channel(params, q0t) # output in t,f
    qzte, qzfe = equalize(params, qzt); # equalized output
    #print(compare(qzte, q0t))
    # demodulation
    shat = demod(params, qzte, ns)
    #shat = demod(params, q0t, ns)
    
    # detection
    stilde = detector(shat, cnt)
    print(stilde==s)
    bhat = symb_to_bit(stilde, cnt)
    print(bhat==b)
 
def question18():
    print("-----------Question 18----------- ") 
    A1 = 1
    A2 = 2
    D = 1
    t0 = 4
    params.L = 1
    #params.sigma2 = 10**(-5)
    q0t = A1*np.exp(-((params.t+t0)**2)/(2*(D**2))) + A2*np.exp(-((params.t-t0)**2)/(2*(D**2))) 
    fig, (ax1, ax2) = plt.subplots(2, 2)
    q1t,_ = channel(params, q0t)
    ax1[0].plot(params.t, q0t, label="time=0")
    ax1[0].plot(params.t, q1t, label="time=1")
    ax1[0].set_ylabel("t0=4")
    ax1[0].legend()
    qzte, qzfe = equalize(params, q1t) # equalized output
    ax1[1].plot(params.t, np.abs(qzte), label="equalized output") # plot input & equalized output in t
    ax1[1].legend()
    t0 = 7
    q0t = A1*np.exp(-((params.t+t0)**2)/(2*(D**2))) + A2*np.exp(-((params.t-t0)**2)/(2*(D**2))) 
    z=1
    q1t,_ = channel(params, q0t)
    ax2[0].plot(params.t, q0t, label="time=0")
    ax2[0].plot(params.t, q1t, label="time=0")
    ax2[0].legend()
    ax2[0].set_ylabel("t0=7")
    qzte, qzfe = equalize(params, q1t) # equalized output
    plt.title("Plot of the input, output and the equalized output", x=0, y=2.2)
    ax2[1].plot(params.t, np.abs(qzte), label="equalized output") # plot input & equalized output in t
    ax2[1].legend()
    plt.show()

def question19(nsnr_sample=50):
    print("-----------Question 19----------- ") 
    #nsnr_sample = 10
    snr_min = 1
    snr_max = 10**6
    BER = np.zeros(nsnr_sample)
    SNR = np.linspace(snr_min, snr_max, nsnr_sample)
    
    fig, (ax1, ax2) = plt.subplots(2, 2)
    snr = SNR[0]
    params.sigma02 = params.P/(snr*params.B*params.L)
    params.sigma2 = (params.sigma02*params.L)/(params.P0*params.T0) #normalized
    BER[0], S1 = Ber(params)
    for i in range(1, nsnr_sample):
        print(i+1,'/',nsnr_sample)
        snr = SNR[i]
        params.sigma02 = params.P/(snr*params.B*params.L)
        params.sigma2 = (params.sigma02*params.L)/(params.P0*params.T0) #normalized
        BER[i], S2 = Ber(params)
    ax1[0].plot(SNR, BER, label="M=2")
    #ax1[1].scatter(S1[0].real, S1[0].imag, color='r', marker='*')
    #ax1[1].scatter(S1[1][:,0], S1[1][:,1], color='r', marker='.')
    ax1[1].scatter(S2[0].real, S2[0].imag, color='g', marker='*')
    ax1[1].scatter(S2[1][:,0], S2[1][:,1], color='r', marker='.')
    ax1[1].set_xlabel("constellation plot at TX and RX for M=2")
    # S = [str(i) for i in range S2]
    S = [str(list(i)) for i in S2[1]]
    tmp = dict(Counter(S))
    for i in tmp:
        res = i.strip('][').split(',')
        v = [float(res[0]), float(res[1])]
        ax1[1].annotate(str(tmp[i]),v, color='r') 
    BER = np.zeros(nsnr_sample)
    SNR = np.linspace(snr_min, snr_max, nsnr_sample)
    BER[0], S1 = Ber(params)
    for i in range(1, nsnr_sample):
        print(i+1,'/',nsnr_sample)
        snr = SNR[i]
        params.sigma02 = params.P/(snr*params.B*params.L)
        params.sigma2 = (params.sigma02*params.L)/(params.P0*params.T0) #normalized
        BER[i], S2 = Ber(params, M=4)
    ax1[0].plot(SNR, BER, label="M=4")  
    ax2[0].scatter(S2[0].real, S2[0].imag, color='g', marker='*')
    ax2[0].scatter(S2[1][:,0], S2[1][:,1], color='r', marker='.') 
    ax2[0].set_xlabel("constellation plot at TX and RX for M=4")
    S = [str(list(i)) for i in S2[1]]
    tmp = dict(Counter(S))
    for i in tmp:
        res = i.strip('][').split(',')
        v = [float(res[0]), float(res[1])]
        ax2[0].annotate(str(tmp[i]),v, color='r') 
    BER[0], S1 = Ber(params) 
    for i in range(1, nsnr_sample):
        print(i+1,'/',nsnr_sample)
        snr = SNR[i]
        params.sigma02 = params.P/(snr*params.B*params.L)
        params.sigma2 = (params.sigma02*params.L)/(params.P0*params.T0) #normalized
        BER[i], S2 = Ber(params, M=16)
        
    ax2[1].scatter(S2[0].real, S2[0].imag, color='g', marker='*')
    ax2[1].scatter(S2[1][:,0], S2[1][:,1], color='r', marker='.')
    
    ax1[0].plot(SNR, BER, label="M=16")
    ax2[1].set_xlabel("constellation plot at TX and RX for M=16")
    S = [str(list(i)) for i in S2[1]]
    tmp = dict(Counter(S))
    for i in tmp:
        res = i.strip('][').split(',')
        v = [float(res[0]), float(res[1])]
        ax2[1].annotate(str(tmp[i]),v, color='r') 
    #ax1[0].axes.xaxis.set_visible(False)
    ax1[0].set_xlabel("SNR")
    ax1[0].set_ylabel("BER")
    fig.suptitle("Plot of the BER as a function of SNR for 2-QAM, 4-QAM and 16-QAM")
    ax1[0].legend()
    plt.show()
    
def question20():
    print("-----------Question 20----------- ") 
    see_alg()

def question21():
    print("-----------Question 21----------- ") 
    see_alg()

if __name__=='__main__':
    params = parameters()
    # params = parameters(_L= 1000000, _B = 10**6, _adb= 0.2, 
    #               _D = 17*(10**(3)), _gama= 1.27*(10**(-3)), 
    #               _lambda0=1.55*(10**(-6)), _P = 6*(10**(-3)),
    #               _T=400, _N=2**10)
    
    pp = parameters(_L= 1000000, _B = 10**6, _adb= 0.2, 
                 _D = 17*(10**(3)), _gama= 1.27*(10**(-3)), 
                 _lambda0=1.55*(10**(-6)), _P = 6*(10**(-3)),
                 _T=400, _N=2**10)
    params.Bn = pp.Bn
    params.sigma2 = pp.sigma2
    params.P0= pp.P0
    params.P= pp.P
    question14()

    
    


## Unit of D

### Question to teacher 
##thats all ?
##What is the predictive model, how to do, pseudo code?
##Question 2
##Equation j 
##sigma2 = f(sigma02)

##Question 11
##What is the question?
##question 13
##question 18


