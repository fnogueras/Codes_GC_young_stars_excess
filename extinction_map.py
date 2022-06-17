#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:05:07 2017

@author: fnoguer
"""


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
import math
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,FixedLocator
import datetime
from scipy.stats.kde import gaussian_kde
import matplotlib as mpl
import scipy.ndimage as ndimage
import scipy
test = 0

tmp = 0 

print(datetime.datetime.now())






name = 'extinction_map'
outdir = ''
nsmooth = 5
radius = 189*0.75
p = 0.25
siz = (29*2)/3
pixscl = 0.053
ini = 0


array = np.loadtxt('')

x_coor,y_coor,J0,dJ,H0,dH,K0,dK = array[:,0], array[:,1], array[:,2], array[:,3], array[:,4], array[:,5], array[:,6], array[:,7]


ax1 = 9701
ax2 = 4818


    
rmap = np.zeros((ax2, ax1))
jkmap = np.zeros((ax2, ax1))
Dexthkmap = np.zeros((ax2, ax1))
mask = np.zeros((ax2, ax1))
  




#############



yes = np.where( (H0 < 99) & (K0 < 99) &  (H0-K0 < 3) & (H0-K0 > 1.3) & (K0 > 13.75) & (K0 < 16.75) )


x_coor = x_coor[yes]
y_coor = y_coor[yes]
J = J0[yes]
H = H0[yes]
K = K0[yes]
dJ = dJ[yes]
dH = dH[yes]
dK = dK[yes]





cmap = plt.cm.jet

x_tmp = x_coor
y_tmp = y_coor


jk = (H-K-0.1) 


mean_acu = []
mean_dm= []
mean_Sigma_HK= []
for xx_index in xrange(5*ini,np.round(ax1/(2*siz))-5*ini):
    for yy_index in xrange(5*ini,np.round(ax2/(2*siz))-5*ini):


        xx = xx_index*(2*siz)
        yy = yy_index*(2*siz)

        yes = np.where( (x_coor > xx-radius) & (x_coor < xx+radius) & (y_coor > yy-radius) & (y_coor < yy+radius) )



        jk_tmp = jk[yes]
        x_used_tmp = x_coor[yes] 
        y_used_tmp = y_coor[yes] 
        
        
        

        d = ((x_used_tmp-xx)**2 + (y_used_tmp-yy)**2)**0.5
        count = np.where(d < radius)
        d = d[count]
        jk_tmp = jk_tmp[count]

        orden = np.argsort(d)

        d = d[orden]                    
        jkvals = jk_tmp[orden]  
 

        if d.size > 1:
            tmp_ref = jkvals[1:]
            d_ref = d[1:]
            
            good = np.where( abs(tmp_ref - jkvals[0]) < 0.3  )    
            
            tmp_ref = tmp_ref[good]
            d_ref = d_ref[good]
            
            
            jkvals = np.insert(tmp_ref,0,jkvals[0])
            d = np.insert(d_ref,0,d[0])
         
        else:
            
            rmap[yy-siz:yy+siz,xx-siz:xx+siz] = 'NaN'                   
            jkmap[yy-siz:yy+siz,xx-siz:xx+siz] = 'NaN'                            
            Dexthkmap[yy-siz:yy+siz,xx-siz:xx+siz] = 'NaN'  
                      
                        

        

        if d.size >= nsmooth:

            
            rad = d[nsmooth-1]
            dist = d[0:nsmooth]
            jkvals = jkvals[0:nsmooth]
            Sigma_Mean = np.std(jkvals)
            num = jkvals/(dist)**p
            den = 1./(dist)**p
            mean_tmp = num.sum()/den.sum()

#Jack Knife uncertainty estimation

            mean_acu = []
            var_tmp = np.empty(dist.size)
            for ii in range(dist.size): 

                if (ii == 0): 


                    dist_tmp = d[1:dist.size]
                    jkvals_tmp = jkvals[1:dist.size]    

     

                if ( (ii != 0) & (ii != dist.size-1) ):


                     dist_tmp1 = d[0:ii]
                     jkvals_tmp1 = jkvals[0:ii]   

                     dist_tmp2 = d[ii+1:dist.size]
                     jkvals_tmp2 = jkvals[ii+1:dist.size]   

                     dist_tmp = np.concatenate([dist_tmp1, dist_tmp2])
                     jkvals_tmp = np.concatenate([jkvals_tmp1,jkvals_tmp2])



                if (ii == dist.size-1):

                    dist_tmp = d[0:dist.size-1]
                    jkvals_tmp = jkvals[0:dist.size-1]    



                num_tmp = jkvals_tmp/(dist_tmp)**p
                den_tmp = 1./(dist_tmp)**p
                mean_tmp1 = num_tmp.sum()/den_tmp.sum()
                mean_tmp1 = np.asarray(mean_tmp1)


                mean_acu.append(mean_tmp1)

                
                var_tmp[ii] = (mean_tmp1-mean_tmp)**2
                
                
            Sigma_JK = max(mean_acu)-min(mean_acu)
            var = ( float((dist.size-1) )/ float(dist.size) ) * sum(var_tmp)
            dm = var**0.5
            

     
            
            
            
            

            A = mean_tmp
            dA = dm

            uncer =  (  ( (1/ (1.84-1 ) )*dA)**2  )**0.5

            rmap[yy-siz:yy+siz,xx-siz:xx+siz] = rad * pixscl                        
            jkmap[yy-siz:yy+siz,xx-siz:xx+siz] = mean_tmp                              
            Dexthkmap[yy-siz:yy+siz,xx-siz:xx+siz] = uncer            
                
        else:
            
            rmap[yy-siz:yy+siz,xx-siz:xx+siz] = 'NaN'                   
            jkmap[yy-siz:yy+siz,xx-siz:xx+siz] = 'NaN'                            
            Dexthkmap[yy-siz:yy+siz,xx-siz:xx+siz] = 'NaN'            
            
print(datetime.datetime.now())


extmap = jkmap/(1.84-1)



 
print(datetime.datetime.now())


hist = extmap.ravel()
good = np.where(hist > 0)
hist = hist[good]
  
 

mask[:,:] = 1 
mask[0:10,:] = 0
mask[:,0:10] = 0
mask[ax2-11:ax2-1,:] = 0
mask[:,ax1-11:ax1-1] = 0


hdu = fits.PrimaryHDU(Dexthkmap*mask)
hdulist = fits.HDUList([hdu])
hdulist.writeto(outdir + 'K_' + name + '_error.fits', clobber=True) 

hdu = fits.PrimaryHDU(extmap*mask)
hdulist = fits.HDUList([hdu])
hdulist.writeto(outdir + 'K_' + name + '.fits', clobber=True) 




print(datetime.datetime.now())



















        
        






