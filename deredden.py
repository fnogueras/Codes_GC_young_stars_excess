#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:28:13 2017

@author: fnoguer
"""


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


def gauss_hist(x, a, x0, sigma, b, x1, sigma1):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + b*np.exp(-(x-x1)**2/(2*sigma1**2))


def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) 

plt.style.use('classic')


deep = 1
cut = 1


out = ''
# Extinction map:
name = ''


# Data
array = np.loadtxt('')
x_coor,y_coor,J,dJ,H,dH,K,dK = array[:,0], array[:,1], array[:,2], array[:,3], array[:,4], array[:,5], array[:,6], array[:,7]
x_coor_ini,y_coor_ini,J_ini,dJ_ini,H_ini,dH_ini,K_ini,dK_ini = array[:,0], array[:,1], array[:,2], array[:,3], array[:,4], array[:,5], array[:,6], array[:,7]





yes = np.where(  (H < 99) & (K < 99) & (H-K > 1.3) )

J_tmp = J

J = J[yes]
K = K[yes]
H = H[yes]



x_coor = x_coor[yes]
y_coor = y_coor[yes]
dJ = dJ[yes]
dK = dK[yes]
dH = dH[yes]



yes = np.where(   (K > -42.86841112520999*(H-K) + 69.72893446277298)  )



J_tmp = J

J = J[yes]
K = K[yes]
H = H[yes]



x_coor = x_coor[yes]
y_coor = y_coor[yes]
dJ = dJ[yes]
dK = dK[yes]
dH = dH[yes]



if deep ==1:
   good = np.where( (H_ini == 99 ) & (K_ini <90) )





   J = np.concatenate((J, J_ini[good]), axis=0)
   K = np.concatenate((K, K_ini[good]), axis=0)
   H = np.concatenate((H, H_ini[good]), axis=0)

   x_coor = np.concatenate((x_coor, x_coor_ini[good]), axis=0)
   y_coor = np.concatenate((y_coor, y_coor_ini[good]), axis=0)
   dK = np.concatenate((dK, dK_ini[good]), axis=0)
   dH = np.concatenate((dH, dH_ini[good]), axis=0)







x = x_coor
y = y_coor




hdulist = fits.open('/extinction_maps/K_' + name)

scidataK = hdulist[0].data
scidataH = scidataK*1.84



scidataJ = hdulist[0].data





AH = scidataH[y.astype(int),x.astype(int)]
AK = scidataK[y.astype(int),x.astype(int)]

yes = np.where( (AK>0)  )              
meanK = np.mean(AK[yes])            
meanH = np.mean(AH[yes])        




yes = np.where((AK>0)  )   



    
K_der = K[yes] - AK[yes]
H_der = H[yes] - AH[yes]




K01 = K_der
H01 = H_der



x = x[yes]
y = y[yes]




H000 = H[yes]
K000 = K[yes]


# Removing overde-reddened stars

if cut ==1:

    good = np.where( (H_der < 80) & (K_der<13.75) & (K_der>12.5) )
    
    cut_mean = np.mean( H_der[good] - K_der[good] )    
    cut_std = np.std( H_der[good] - K_der[good] )
    
    
    good = np.where( H_der>80 )
    
    K_add = K_der[good]
    H_add = H_der[good]

    
    x_add = x[good]
    y_add = y[good]
    
    
    


    
    yes = np.where(   ( (H_der-K_der)>cut_mean-2*cut_std ) )
    
    
    x = x[yes]
    y = y[yes]

    H_der = H_der[yes]
    K_der = K_der[yes] 
    
    
    
    
    
    np.savetxt(out + 'de_reddened.txt', np.c_[x,y,H_der,K_der,H000[yes],K000[yes]])

