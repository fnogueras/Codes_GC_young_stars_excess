#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:01:31 2019

@author: fnoguer
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d




# Completeness solution
array = np.loadtxt('')
mag, com, err = array[:,0], array[:,1], array[:,2]


# De-reddened stars to compute median extincion and apply it to the completeness solution
array = np.loadtxt('')
x0,y0,J0,H0,K0,H,K = array[:,0], array[:,1], array[:,2], array[:,3], array[:,4], array[:,5], array[:,6]      
median_ext = np.median(K-K0)




mag = mag - median_ext

f = interp1d(mag, com)
f_err = interp1d(mag, err)


f_inv = interp1d(com, mag)

xnew = np.linspace(min(mag), max(mag), num=410, endpoint=True)
     


test = f_inv(0.75)


# KLF to be corrected for completeness
array = np.loadtxt('')
x_cen0, y0 = array[:,0], array[:,1]


good = np.where( (x_cen0 > min(xnew) ) & ((x_cen0 < test) ) )



y_corr = y0[good]




# Checking plots
     
fig, ax = plt.subplots(1,1, figsize=(6, 6), facecolor='w', edgecolor='k')
ax.plot(x_cen0[good], f(x_cen0[good]), label='fit',linewidth=2.0)  



    
fig, ax = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k')
ax.errorbar(x_cen0,y0,yerr=y0**0.5, linestyle='-', fmt='.', alpha = 0.2)
ax.set_yscale("log")
ax.set_xlabel('Ks', fontsize=20)
ax.set_ylabel('# stars', fontsize=20)
ax.axis([8.8, 18., 1, 5*max(y0)])



y_fin = y_corr + y_corr * (1-f(x_cen0[good]) )
ax.errorbar(x_cen0[good],y_fin,yerr=y_fin**0.5, linestyle='-', fmt='.', alpha = 0.2)





yes = np.where(x_cen0 < min(x_cen0[good]))

x_fin0 = x_cen0[yes]
y_fin0 = y0[yes]


y_err = y0[yes]**0.5
          
          
          
yes = np.where( (x_cen0 >= min(x_cen0[good])) & (x_cen0 <test)  )

# Estimating the uncertainty
y_err1 = ( ((2-f(x_cen0[yes]) )* y0[yes]**0.5)**2 + (y0[yes]*f_err(x_cen0[yes]) )**2  )**0.5


x_fin1 = x_cen0[good]
y_fin1 = y_fin





x_final =  np.concatenate((x_fin0, x_fin1), axis=0)
y_final =  np.concatenate((y_fin0, y_fin1), axis=0)

y_err_final = np.concatenate((y_err, y_err1), axis=0)
 
    
fig, ax = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k')

ax.errorbar(x_final,y_final,yerr=y_err_final, linestyle='-', fmt='.', alpha = 0.2)
ax.set_yscale("log")
ax.set_xlabel('Ks', fontsize=20)
ax.set_ylabel('# stars', fontsize=20)
ax.axis([7, 18., 1, 5*max(y0)])


ax.errorbar(x_cen0,y0,yerr=y0**0.5, linestyle='-', fmt='.', alpha = 0.2)

# Corrected KLF
np.savetxt('', np.c_[x_final,y_final,y_err_final])


























