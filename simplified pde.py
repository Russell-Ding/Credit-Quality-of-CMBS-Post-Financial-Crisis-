# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 01:21:57 2016

@author: Russell
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize as mmiz 


def loan_pde(vol_p,C,q,Y,p0,r0):
    '''
    
    The function uses finite difference method to derive the initial loan value
    based on Titman & Torous (1989) model pde.
    
    Boundary Condition:
    1. At maturity T, the loan value is 0
    2. Default happens when housr price dropped beow loan vaue.
    When default, the house price face an immediate deduction of 24.4%,
    which is equivalent to the loan value will never exceed 75.6% of the property price
    
    
    q: Realized income return, varies with property type
    
    C: Monthly Coupon Payment, WAC here
    
    Y: time (Year)
    
    vol_p: volatility of the housing price
    
    p0: Intial Property Price, can be derived by ltv and original balance, measured in million
    
    r0: Market zero Rate
    
    N: Totla number of months to count back
    
    The function solve the pde and return the initial value of the loan 
    under the assumption that the inital value of the property os 1 dollar
    '''
    kappa=0.13131
    theta=0.0574
    vol_r=0.06035
    eta=-0.07577
    
    
    N=25
    dt=Y/float(N)    
    #plot the gradian
    dr=np.sqrt(vol_r**2*dt)+0.0001
    dp=np.sqrt(vol_p**2*dt)+0.01
    
    
   
    R0=pd.Series(np.ones(2*N+1))*dr+(r0-N*dr)
    P0=pd.Series(np.ones(2*N+1))*dp+(p0-N*dp)
    
    #Initial Boundary condition, the loan has zero value at marutiy
    #Going down from the row is the M value with interest rate
    #Going right to the columns is the M value for different propoerty price
    old_M=pd.DataFrame(0,index=range(0,2*N+1),columns=range(0,2*N+1))
    
    #Calculate backward to obtain the Mortgate price at t=0
    i=2*N
    j=1
    
    new_M=pd.DataFrame()
    while (j<=i):
        
        new_M=pd.DataFrame(0,index=range(j,i),columns=range(j,i))
        for row in range(j,i):
            for column in range(j,i):
                #weight for M(i,j,k+1)
                weigth1=1-R0.loc[row]*dt-vol_r**2*R0.loc[row]*dt/dr**2
                #weight for M(i+1,j,k+1)
                weight2=R0.loc[row]*dt/2*vol_r**2/dr**2+dt*(kappa*(theta-R0.iloc[row])-eta*R0.loc[row])/(2*dr)
                #weight for M(i-1,j,k+1)
                weight3=R0.loc[row]*dt/2*vol_r**2/dr**2+dt*(kappa*(theta-R0.iloc[row])-eta*R0.loc[row])/(2*dr)*(-1)
                #weight for M(i,j,k+1)
                weight4=-P0.loc[column]**2*dt*vol_p**2/dp**2
                #weight for M(i,j+1,k+1)
                weight5=dt/2*vol_p**2/dp**2*P0.loc[column]**2-(R0.loc[row]-q)*P0.loc[column]/(2*dp)*dt
                #weight for M(i,j-1,k+1)
                weight6=dt/2*vol_p**2/dp**2*P0.loc[column]**2+(R0.loc[row]-q)*P0.loc[column]/(2*dp)*dt
                #Assign value to matrix at dt forward
                new_M.loc[row,column]=dt*C+weigth1*old_M.loc[row,column]+\
                +weight2*old_M.loc[row+1,column]\
                +weight3*old_M.loc[row-1,column]\
                +weight4*old_M.loc[row,column]\
                +weight5*old_M.loc[row,column+1]\
                +weight6*old_M.loc[row,column-1]
                #Adjusting the value for default
                #In case loan price M is larger than house price, loan default and will be adjust to 0.756 of house price
                if (new_M.loc[row,column]>P0.loc[column]):
                    new_M.loc[row,column]=P0.loc[column]*0.756
        old_M=new_M
        j=j+1
        i=i-1
    return new_M
   
def second_norm(vol_p,C,q,Y,p0,r0,target_price):
    temp=loan_pde(vol_p,C,q,Y,p0,r0)
    error=(temp.iloc[0,0]-target_price)**2
    
    return error

raw=pd.read_excel('implied_vol_skeleton.xlsx')
raw['implied_vol']=0

for i in range(0,len(raw.index.values)):
    vol_p=0.3
    res=mmiz(second_norm,vol_p,args=(raw['Coupon'].iloc[i]*raw['Loan Balance'].iloc[i],raw['q'].iloc[i]\
    ,raw['Time'].iloc[i],raw['House_Price'].iloc[i],raw['rf'].iloc[i],raw['Loan Balance'].iloc[i]),method='TNC',bounds=((0,0.7),))
    raw['implied_vol'].iloc[i]=res.x[0]

writer = pd.ExcelWriter('sum_implied_vol(Retail).xlsx')
raw.to_excel(writer,'Sheet1')
writer.save()

