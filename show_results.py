#!/usr/bin/env python




import ALpython as AL
import ALCarpLib as ALC
import matplotlib.pyplot as plt
import numpy as np
import os
from carputils.carpio import igb
import pandas as pd
import scipy.optimize
import itertools



if __name__ == "__main__":

   

    Trefref1 = 15
    Trefnec = np.array([0, 15, 20]) #np.arange(0,40,5)

    Ffinal = [None] * len(Trefnec)
    fig, ax = plt.subplots(ncols=2, nrows=2)
    marker = itertools.cycle(('o', 's', 'v', '^', '+', '*')) 

    for i1, Trefnec1 in enumerate(Trefnec):
        DataDir = f'out/20210707/fracnec0.1_Trefref{Trefref1}_Trefnec{Trefnec1}'

        if os.path.exists(DataDir+'/basal_force.dat'):
            F = np.loadtxt(DataDir+'/basal_force.dat')
            ax[0,0].plot(F[:,0], F[:,1], label=f"Trefref={Trefref1}, Trefnec={Trefnec1}")
            ax[0,0].set_ylabel('basal force')
            ax[0,0].set_xlabel('time')
            ax[0,0].legend(loc=(0))
            Ffinal[i1] = F[-1,1]

        T = igb.IGBFile(DataDir+'/Tension.igb').data()
        print('len(T)=',len(T))
        ax[1,0].plot(T) #[1:len(T.data()):101+10*i1],'.')
        ax[1,0].set_ylabel('Tension.igb')

        S = igb.IGBFile(DataDir+'/stressTensor.igb').data()
        S = np.reshape(S, (-1,9))
        print('len(S)=',len(S))
        trS = [None] * len(S)
        for js1, s1 in enumerate(S):
            trS[js1] = np.trace(np.reshape(s1, (3,3)))
        ax[1,1].plot(trS)
        ax[1,1].set_ylabel('trace (stressTensor.igb) ')



    ax[0,1].plot(Trefnec, Ffinal, '.-')
    ax[0,1].set_ylabel('Ffinal')
    ax[0,0].set_xlabel('Trefnec')


    ax[1,0].set_xlim((42900,42980))
    ax[1,1].set_xlim((151000,158000))
    plt.show()