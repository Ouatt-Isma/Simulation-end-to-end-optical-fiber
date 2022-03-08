import math 
import numpy as np
class parameters:
    def __init__(self, _L= 1000000, _B = 10**10, _adb= 0.2, 
                 _D = 17*(10**(-6)), _gama= 1.27*(10**(-3)), 
                 _lambda0=1.55*(10**(-6)), _P = 6*(10**(-3)),
                 _T=400, _N=2**10):
        #variables
        self.L = _L #distance
        self.B = _B #Bandwidth
        #Physical constant
        self.adb = _adb
        self.D = _D # dispersion ps/(nm-km)
        self.gama = 1.27*(10**(-3)) # nonlinearity coefficient
        nsp = 1 # a constant factor
        h = 6.626*(10**(-34)) # Planck constant
        self.lambda0 = _lambda0 #center wavelength
        c=3*(10**8) #celerity
        self.f0 = c/self.lambda0 #center frequency
        self.alpha =  ((10**(-4))*math.log10(10))*self.adb #loss coefficient
        self.beta2 = -((self.lambda0**2)*self.D)/(2*np.pi*c) #dispersion coefficient
        #scale factors
        self.L0 = self.L
        self.T0 = math.sqrt((np.abs(self.beta2)*self.L)/2)
        self.P0 = 2/(self.gama*self.L)
        self.Bn = self.B*self.T0
        #noise PSD
        self.sigma02 = nsp*h*self.alpha*self.f0 #physical
        self.sigma2 = (self.sigma02*self.L)/(self.P0*self.T0) #normalized
        #Signal power
        self.P = _P/self.P0
        #####
        # time mesh
        self.T = _T# you have to choose this
        self.N = _N # you have to choose this
        self.dt = self.T/(self.N-1) #T/N
        self.t = np.linspace(-self.T/2, self.T/2, self.N) #should be from -T/2 to T/2 with length N
        # frequency mesh
        self.F = 1/self.dt #
        self.df = 1/self.T #
        self.f =np.linspace(-self.F/2, self.F/2, self.N) # should be from -F/2 to F/2 with length N
 