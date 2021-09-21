# -*- coding: utf-8 -*-
"""lp0;'

Created on Sun Dec 27 07:35:35 2020

@author: MOHAMMAD SHARIQUE AHMAD
ahmad@eurecom.fr
eshariq.am@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import numpy.matlib as nm

SNRdB = np.arange(0,11,2);
ITER = 1000;
Nt = 4;
Nr = 4;
M = 2; #BPSK Modulation
bpcu = np.log2(M*Nt);
BERopt = np.zeros(len(SNRdB));
for ite in range(ITER):
  #  print (ite);
    isym = nr.randint(2**bpcu);
    antIndex = isym % 4;
    sym = 2*(isym>3)-1;
    H = 1/np.sqrt(2)*(nr.normal(0,1,(Nr,Nt)) + 1j*nr.normal(0,1,(Nr,Nt)));
    RxNoise = 1/np.sqrt(2)*(nr.normal(0,1,(Nr,1)) + 1j*nr.normal(0,1,(Nr,1)));
    for K in range(len(SNRdB)):
        rho = 10**(SNRdB[K]/10);
        RxVec = np.sqrt(rho)*H[:,antIndex:antIndex+1]*sym + RxNoise;
        
        MLobj=np.sum(np.absolute(np.sqrt(rho)*np.concatenate((-H,H),axis=1)-nm.repmat(RxVec,1,2*Nt))**2,axis=0);
        decIndex = np.argmin(MLobj);
        BERopt[K] = BERopt[K] + (decIndex != isym);
        

BERopt = BERopt/(ITER);

plt.yscale('log')
plt.plot(SNRdB, BERopt,'gs-');
plt.grid(1,which='both')
plt.suptitle('BER for SM')
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 

