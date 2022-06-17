#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:50:08 2021

@author: nogueras
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import datetime
mpl.style.use('classic')


print_screen = 1

a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 =  '14', '11', '8', '6','3','1.5','0600', '0400', '0200','0040','0020','0010','0005','0002'
met1 = '2Z'


# Arrays with the result for each MC iteration

age1_total = np.empty(1)
age2_total = np.empty(1)
age3_total = np.empty(1)
age4_total = np.empty(1)          
age5_total = np.empty(1)
age6_total = np.empty(1)
age7_total = np.empty(1)
age8_total = np.empty(1)
age9_total = np.empty(1)
age10_total = np.empty(1)
age11_total = np.empty(1)
age12_total = np.empty(1)
age13_total = np.empty(1)
age14_total = np.empty(1)

chi_total = np.empty(1)
smooth_total = np.empty(1)
mod_total = np.empty(1)
tot = np.empty(1)



for i in range(1):
    print(datetime.datetime.now())

    array = np.loadtxt('')
   

    x_cen0, y0, y_err = array[:,0], array[:,1], array[:,2]
                
    size = 0.1
                
                
    complete = np.where(y_err == 0)
    
    y_err[complete] = 1

    
    
    
    
    good = np.where( (x_cen0 > 8.5) & (x_cen0 < 16.5) )
    
    
    
    x_cen0 , y0, y_uncer = x_cen0[good] , y0[good] , y_err[good]
    
    
    def fun(x_fit, a,b,d,e,ff,ee,fff,ffff,fffff,c,sigma1,ffffff,aa,aaa,aaaa,aaaaa):
        
        
        
    
        
        
        array = np.loadtxt('/KLF_3.6/' + str(a1) + '_' + met1 + '.txt')
        
        K, logn = array[:,2], array[:,8]
    
        x = K + c
        y = logn
        
        xnew =np.linspace(5,19,num = 410,endpoint = True)
        
        
        f2 = interp1d(x, y, kind='cubic')
        
        y2 = gaussian_filter1d(f2(xnew),sigma1)
        f2 = interp1d(xnew,y2,kind = 'cubic')
        
        
        ##########
        
        array = np.loadtxt('/KLF_3.6/' + str(a2) + '_' + met1 + '.txt')
        
        K, logn = array[:,2], array[:,8]
    
        x0 = K + c
        y0 = logn
        
        
    
        
        f = interp1d(x0, y0)
        f3 = interp1d(x0, y0, kind='cubic')
        
        y2 = gaussian_filter1d(f3(xnew),sigma1)
        f3 = interp1d(xnew,y2,kind = 'cubic')    
        
    
     ##########
                    
        array = np.loadtxt('/KLF_3.6/' + str(a3) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x1 = K + c
        y1 = logn
                    
                    
                
                    
        f = interp1d(x1, y1)
        f4 = interp1d(x1, y1, kind='cubic')
    
        y2 = gaussian_filter1d(f4(xnew),sigma1)
        f4 = interp1d(xnew,y2,kind = 'cubic')   
                
         ##########
         
         
         
        array = np.loadtxt('/KLF_3.6/' + str(a4) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x2 = K + c
        y2 = logn
                    
                    
                
                    
        f = interp1d(x2, y2)
        f5 = interp1d(x2, y2, kind='cubic')
    
        y2 = gaussian_filter1d(f5(xnew),sigma1)
        f5 = interp1d(xnew,y2,kind = 'cubic')                   
         ##########     
         
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a5) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x3 = K + c
        y3 = logn
                    
                    
                
                    
        f = interp1d(x3, y3)
        f6 = interp1d(x3, y3, kind='cubic')
    
        y2 = gaussian_filter1d(f6(xnew),sigma1)
        f6 = interp1d(xnew,y2,kind = 'cubic')       
    
    
         ##########     
         
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a6) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x4 = K + c
        y4 = logn
                    
                    
                
                    
        f = interp1d(x4, y4)
        f7 = interp1d(x4, y4, kind='cubic')
    
        y2 = gaussian_filter1d(f7(xnew),sigma1)
        f7 = interp1d(xnew,y2,kind = 'cubic')       
    
         ##########     
         
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a7) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x5 = K + c
        y5 = logn
                    
                    
                
                    
        f = interp1d(x5, y5)
        f8 = interp1d(x5, y5, kind='cubic')
    
        y2 = gaussian_filter1d(f8(xnew),sigma1)
        f8 = interp1d(xnew,y2,kind = 'cubic')       
                
         ##########      
         
         
         
        array = np.loadtxt('/KLF_3.6/' + str(a8) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x6 = K + c
        y6 = logn
                    
                    
                
                    
        f = interp1d(x6, y6)
        f9 = interp1d(x6, y6, kind='cubic')
    
        y2 = gaussian_filter1d(f9(xnew),sigma1)
        f9 = interp1d(xnew,y2,kind = 'cubic')       
                
         ##########   
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a9) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x7 = K + c
        y7 = logn
                    
                    
                
                    
        f = interp1d(x7, y7)
        f10 = interp1d(x7, y7, kind='cubic')    
             
    
        y2 = gaussian_filter1d(f10(xnew),sigma1)
        f10 = interp1d(xnew,y2,kind = 'cubic')              
        
         ##########   
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a10) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x8 = K + c
        y8 = logn
                    
                    
                
                    
        f = interp1d(x8, y8)
        f11 = interp1d(x8, y8, kind='cubic')    
             
        y2 = gaussian_filter1d(f11(xnew),sigma1)
        f11 = interp1d(xnew,y2,kind = 'cubic')       
           
          ##########   
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a11) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x9 = K + c
        y9 = logn
                    
                    
                
                    
        f = interp1d(x9, y9)
        f12 = interp1d(x9, y9, kind='cubic')        
        
        y2 = gaussian_filter1d(f12(xnew),sigma1)
        f12 = interp1d(xnew,y2,kind = 'cubic')           
        
          ##########   
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a12) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x10 = K + c
        y10 = logn
                    
                    
                
                    
        f = interp1d(x10, y10)
        f13 = interp1d(x10, y10, kind='cubic') 
    
        y2 = gaussian_filter1d(f13(xnew),sigma1)
        f13 = interp1d(xnew,y2,kind = 'cubic')       
        
          ##########   
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a13) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x11 = K + c
        y11 = logn
                    
                    
                
                    
        f = interp1d(x11, y11)
        f14 = interp1d(x11, y11, kind='cubic')  

        y2 = gaussian_filter1d(f14(xnew),sigma1)
        f14 = interp1d(xnew,y2,kind = 'cubic')   
    
          ##########   
         
         
          
        array = np.loadtxt('/KLF_3.6/' + str(a14) + '_' + met1 + '.txt')
        
                    
        K, logn = array[:,2], array[:,8]
         
        x12 = K + c
        y12 = logn
                    
                    
                
                    
        f = interp1d(x12, y12)
        f15 = interp1d(x12, y12, kind='cubic')  

        y2 = gaussian_filter1d(f15(xnew),sigma1)
        f15 = interp1d(xnew,y2,kind = 'cubic')   
                    
        final = a*f2(x_fit)+b*f3(x_fit)+d*f4(x_fit)+e*f5(x_fit)+ff*f6(x_fit) +ee*f7(x_fit)+fff*f8(x_fit)+ffff*f9(x_fit)+fffff*f10(x_fit)+ffffff*f11(x_fit) + aa * f12(x_fit)+ aaa * f13(x_fit) + aaaa*f14(x_fit) +aaaaa*f15(x_fit)
       

    
        return final
   
    
    popt, pcov = curve_fit(fun, x_cen0, y0, sigma = y_uncer,p0 = [10,10,10,10,10,10,10,10,10,14.6,5,10,10,10,10,10], bounds=([0.,0,0,0,0,0,0,0,0, 14.15,5,0,0,0,0,0], [30000000,30000000,30000000,30000000,30000000,30000000,30000000,30000000,30000000, 15.05, 15,30000000,30000000,30000000,30000000,30000000]) )    #popt, pcov = curve_fit(fun, x_cen0, y0, bounds=([0.,0., 0.,0.91, 14.35, 7], [10.,10.,10.,1.30, 14.85, 34]) )  
    
    #Derived Chi Squared Value For This Model
     
    chi_squared = np.sum(((fun(x_cen0, *popt)-y0)/y_uncer)**2)
    reduced_chi_squared = (chi_squared)/(len(x_cen0)-len(popt))
    print 'The degrees of freedom for this test is', len(x_cen0)-len(popt)
    print 'The chi squared value is: ',("%.2f" %chi_squared)
    print 'The reduced chi squared value is: ',("%.2f" %reduced_chi_squared)
    
    


    
    mod_total[i] = popt[9]
    smooth_total[i] = popt[10]  
    chi_total[i] = reduced_chi_squared    
    
    tot[i] = popt[0] + popt[1] + popt[2] + popt[3] +popt[4] + popt[5] +popt[6] + popt[7] +popt[8] + popt[11] +popt[12] + popt[13] +popt[14]+popt[15]

    age1_total[i] = popt[0]/tot[i]
    age2_total[i] = popt[1]/tot[i]
    age3_total[i] = popt[2]/tot[i]
    age4_total[i] =  popt[3]/tot[i]           
    age5_total[i] = popt[4]/tot[i]
    age6_total[i] = popt[5]/tot[i]
    age7_total[i] = popt[6]/tot[i]
    age8_total[i] =  popt[7]/tot[i]   
    age9_total[i] = popt[8]/tot[i]
    age10_total[i] = popt[11]/tot[i]
    age11_total[i] = popt[12]/tot[i]
    age12_total[i] = popt[13]/tot[i]    
    age13_total[i] = popt[14]/tot[i]
    age14_total[i] = popt[15]/tot[i]
#########


# Plot


xnew=np.linspace(5,19,num = 410,endpoint = True)

sigma1 = popt[10]
    
    
array = np.loadtxt('/KLF_3.6/' + str(a1) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x = K + popt[9]
y = logn
    

    
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')
    
y2 = gaussian_filter1d(f2(xnew),sigma1)
f2 = interp1d(xnew,y2,kind = 'cubic')  
    
    
    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a2) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x00 = K + popt[9]
y00 = logn
    
    

    
f = interp1d(x00, y00)
f3 = interp1d(x00, y00, kind='cubic')

y2 = gaussian_filter1d(f3(xnew),sigma1)
f3 = interp1d(xnew,y2,kind = 'cubic')  
    
        

     ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a3) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x000 = K + popt[9]
y000 = logn
    
    

    
f = interp1d(x000, y000)
f4 = interp1d(x000, y000, kind='cubic')   

y2 = gaussian_filter1d(f4(xnew),sigma1)
f4 = interp1d(xnew,y2,kind = 'cubic') 

###########


    
array = np.loadtxt('/KLF_3.6/' + str(a4) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x2 = K + popt[9]
y2 = logn
    

    
f = interp1d(x2, y2)
f5 = interp1d(x2, y2, kind='cubic')
    

y2 = gaussian_filter1d(f5(xnew),sigma1)
f5 = interp1d(xnew,y2,kind = 'cubic')     
    
    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a5) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x3 = K + popt[9]
y3 = logn
    
    

    
f = interp1d(x3, y3)
f6 = interp1d(x3, y3, kind='cubic')
    
y2 = gaussian_filter1d(f6(xnew),sigma1)
f6 = interp1d(xnew,y2,kind = 'cubic') 


###########


    
array = np.loadtxt('/KLF_3.6/' + str(a6) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x4 = K + popt[9]
y4 = logn
    

    
f = interp1d(x4, y4)
f7 = interp1d(x4, y4, kind='cubic')
    
y2 = gaussian_filter1d(f7(xnew),sigma1)
f7 = interp1d(xnew,y2,kind = 'cubic') 
    
    
    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a7) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x5 = K + popt[9]
y5 = logn
    
    

    
f = interp1d(x5, y5)
f8 = interp1d(x5, y5, kind='cubic')


y2 = gaussian_filter1d(f8(xnew),sigma1)
f8 = interp1d(xnew,y2,kind = 'cubic') 
    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a8) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x6 = K + popt[9]
y6 = logn
    
    

    
f = interp1d(x6, y6)
f9 = interp1d(x6, y6, kind='cubic')

y2 = gaussian_filter1d(f9(xnew),sigma1)
f9 = interp1d(xnew,y2,kind = 'cubic') 

    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a9) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x7 = K + popt[9]
y7 = logn
    
    

    
f = interp1d(x7, y7)
f10 = interp1d(x7, y7, kind='cubic')

y2 = gaussian_filter1d(f10(xnew),sigma1)
f10 = interp1d(xnew,y2,kind = 'cubic') 

    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a10) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x8 = K + popt[9]
y8 = logn
    
    

    
f = interp1d(x8, y8)
f11 = interp1d(x8, y8, kind='cubic')

y2 = gaussian_filter1d(f11(xnew),sigma1)
f11 = interp1d(xnew,y2,kind = 'cubic') 

    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a11) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x9 = K + popt[9]
y9 = logn
    
    

    
f = interp1d(x9, y9)
f12 = interp1d(x9, y9, kind='cubic')

y2 = gaussian_filter1d(f12(xnew),sigma1)
f12 = interp1d(xnew,y2,kind = 'cubic') 


    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a12) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x10 = K + popt[9]
y10 = logn
    
    

    
f = interp1d(x10, y10)
f13 = interp1d(x10, y10, kind='cubic')



    ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a13) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x11 = K + popt[9]
y11 = logn
    
    

    
f = interp1d(x11, y11)
f14 = interp1d(x11, y11, kind='cubic')
  
y2 = gaussian_filter1d(f14(xnew),sigma1)
f14 = interp1d(xnew,y2,kind = 'cubic')   
        ##########
    
array = np.loadtxt('/KLF_3.6/' + str(a14) + '_' + met1 + '.txt')
    
K, logn = array[:,2], array[:,8]

x12 = K + popt[9]
y12 = logn
    
    

    
f = interp1d(x12, y12)
f15 = interp1d(x12, y12, kind='cubic')


y2 = gaussian_filter1d(f15(xnew),sigma1)
f15 = interp1d(xnew,y2,kind = 'cubic') 
    
#final = gaussian_filter1d(popt[0]*f2(x_cen0) + popt[1]*f3(x_cen0) + popt[2]*f4(x_cen0)  + popt[3]*f5(x_cen0) + popt[4]*f6(x_cen0) ,popt[5])

    
final = popt[0]*f2(x_cen0) + popt[1]*f3(x_cen0) + popt[2]*f4(x_cen0) + popt[3]*f5(x_cen0)+ popt[4]*f6(x_cen0) + popt[5]*f7(x_cen0)+ popt[6]*f8(x_cen0)+ popt[7]*f9(x_cen0)+ popt[8]*f10(x_cen0)+ popt[11]*f11(x_cen0)+ popt[12]*f12(x_cen0)+ popt[13]*f13(x_cen0)+ popt[14]*f14(x_cen0)+ popt[15]*f15(x_cen0)


print(datetime.datetime.now())

fig, ax = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k')



ax.errorbar(x_cen0,y0,yerr=y_uncer, linestyle='-', fmt='.', alpha = 0.1)
ax.plot(x_cen0, final, label='fit',linewidth=2, linestyle='--', color = 'red', alpha = 1)

ax.plot(xnew, popt[0]*f2(xnew), label='fit',linewidth=2.0, color = 'orange', alpha = 0.5)
ax.plot(xnew, popt[1]*f3(xnew), label='fit',linewidth=2.0, color = 'green', alpha = 0.5)
ax.plot(xnew, popt[2]*f4(xnew), label='fit',linewidth=2.0, color = 'purple', alpha = 0.5)
ax.plot(xnew, popt[3]*f5(xnew), label='fit',linewidth=2.0, color = 'cyan', alpha = 0.5)
ax.plot(xnew, popt[4]*f6(xnew), label='fit',linewidth=2.0, color = 'blue', alpha = 0.5)
ax.plot(xnew, popt[5]*f7(xnew), label='fit',linewidth=2.0, color = 'cyan', alpha = 0.5)
ax.plot(xnew, popt[6]*f8(xnew), label='fit',linewidth=2.0, color = 'blue', alpha = 0.5)
ax.plot(xnew, popt[7]*f9(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)
ax.plot(xnew, popt[8]*f10(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)

ax.plot(xnew, popt[11]*f11(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)
ax.plot(xnew, popt[12]*f12(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)
ax.plot(xnew, popt[13]*f13(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)
ax.plot(xnew, popt[14]*f14(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)

ax.plot(xnew, popt[15]*f15(xnew), label='fit',linewidth=2.0, color = 'red', alpha = 0.5)


ax.set_yscale("log")
ax.axis([6, 17, 1, 100000])


red = format(float(reduced_chi_squared), '.2f')    
ax.annotate(r'$\chi^2$' + ' = '  + ' ' + str(red) , xy=(0.38, 0.94), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='top')   
ax.set_xlabel('Ks', fontsize=20)
ax.set_ylabel('# stars', fontsize=20)


ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)



fig1, ax1 = plt.subplots(1,1, figsize=(7, 5), facecolor='w', edgecolor='k', sharey = True)
plt.gcf().subplots_adjust(bottom=0.15, left = 0.2, right = 0.7, top = 0.88,wspace = 0.125, hspace = 0.125)



ax1.errorbar(x_cen0,y0,yerr=y_uncer, linestyle='-', fmt='.', alpha = 0.35, label = 'data', color = 'steelblue')

ax1.plot(x_cen0, final,linewidth=2, linestyle='--', color = 'red', alpha = 1, label = 'fit')


ax1.plot(xnew, popt[0]*f2(xnew) + popt[1]*f3(xnew) + popt[2]*f4(xnew),linewidth=2.0, color = 'orange', alpha = 0.5, label = '> 7 Gyr')


ax1.plot(xnew,   popt[3]*f5(xnew) + popt[4]*f6(xnew),linewidth=2.0, color = 'green', alpha = 0.5, label = '2-7 Gyr')

ax1.plot(xnew,  popt[5]*f7(xnew) + popt[6]*f8(xnew),linewidth=2.0, color = 'blue', alpha = 0.5, label = '0.5-2 Gyr')

ax1.plot(xnew,  popt[7]*f9(xnew) + popt[8]*f10(xnew) + popt[11]*f11(xnew),linewidth=2.0, color = 'violet', alpha = 0.7, label = '0.08-0.5 Gyr')

ax1.plot(xnew, popt[12]*f12(xnew) + popt[13]*f13(xnew) + popt[14]*f14(xnew) + popt[15]*f15(xnew),linewidth=2.0, color = 'purple', alpha = 0.7, label = '0-0.08 Gyr')



ax1.set_yscale("log")
ax1.axis([8.5, 16.1, 1, 100000])


red = format(float(reduced_chi_squared), '.2f')    
ax1.annotate(r'$\chi^2$' + ' = '  + ' ' + str(red) , xy=(0.38, 0.94), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='top')   
ax1.set_xlabel('Ks', fontsize=20)
ax1.set_ylabel('# stars', fontsize=20)


ax1.tick_params(axis='x', labelsize=16)
ax1.tick_params(axis='y', labelsize=16)

ax1.legend(bbox_to_anchor = (1.4,1.1), shadow=True, fancybox=True, prop={'size': 12}, ncol=2)

# fig1.savefig('KLF_SgrB1.pdf')

if print_screen == 1:

    print(a1 + ' -> ' + str(format(float(100*popt[0]/tot), '.2f')  ) )
    
    print(a2 + ' -> ' + str(format(float(100*popt[1]/tot), '.2f')  ) )
    
    print(a3 + ' -> ' + str(format(float(100*popt[2]/tot), '.2f')  ) )
    
    print(a4 + ' -> ' + str(format(float(100*popt[3]/tot), '.2f')  ) )
    
    print(a5 + ' -> ' + str(format(float(100*popt[4]/tot), '.2f')  ) )
    
    print(a6 + ' -> ' + str(format(float(100*popt[5]/tot), '.2f')  ) )
    
    print(a7 + ' -> ' + str(format(float(100*popt[6]/tot), '.2f')  ) )
    
    print(a8 + ' -> ' + str(format(float(100*popt[7]/tot), '.2f')  ) )
    
    print(a9 + ' -> ' + str(format(float(100*popt[8]/tot), '.2f')  ) )
    
    print(a10 + ' -> ' + str(format(float(100*popt[11]/tot), '.2f')  ) )
    
    print(a11 + ' -> ' + str(format(float(100*popt[12]/tot), '.2f')  ) )
    
    print(a12 + ' -> ' + str(format(float(100*popt[13]/tot), '.2f')  ) )
    
    print(a13 + ' -> ' + str(format(float(100*popt[14]/tot), '.2f')  ) )
    
    print(a14 + ' -> ' + str(format(float(100*popt[15]/tot), '.2f')  ) )





