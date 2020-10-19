---
layout: post
title: "Malaria Model"
date: 2019-10-07
---

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def sh(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M):
  return (-delta*sh*im/N) + alpha*ih

def ih(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M):
  return -beta*ih - gamma*ih - alpha*ih + delta*sh*im/M

def imh(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M):
  return beta*ih

def sm(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M):
  return kappa*M - delta*sm*ih/N - eta*sm

def im(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M):
  return delta*sm*ih/N - eta*im
```


```python
start_time = 0
end_time = 1000
dt = 2
n = int((end_time - start_time)/dt)

Sh = np.zeros(n) 
Ih = np.zeros(n) 
Imh = np.zeros(n) 
Sm = np.zeros(n) 
Im = np.zeros(n) 

Sh[0] = 300
Ih[0] = 1
Imh[0] = 0
Sm[0] = 300
Im[0] = 0

alpha = 0.3
beta = 0.01
gamma = 0.005
kappa = 0.01
delta = 0.3
eta = 0.01

N = Sh[0] + Ih[0] + Imh[0]
M = Sm[0] + Im[0]

for i in range(1,len(Sh)):
  Sh[i] = Sh[i-1] + dt*sh(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M)
  Ih[i] = Ih[i-1] + dt*ih(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M)
  Imh[i] = Imh[i-1] + dt*imh(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M)
  Sm[i] = Sm[i-1] + dt*sm(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M)
  Im[i] = Im[i-1] + dt*im(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M)
  
plt.plot(np.arange(start_time,end_time,dt),Sh/N,color='green',label='Susceptible Humans')  
plt.plot(np.arange(start_time,end_time,dt),Ih/N,color='blue',label='Infected Humans')  
plt.plot(np.arange(start_time,end_time,dt),Imh/N,color='red',label='Immune Humans')  
plt.legend()
plt.xlabel('Time (in days)')
plt.ylabel('Fraction of humans')
plt.title('Fraction of humans of every category (Basic Model)')
plt.show()
plt.plot(np.arange(start_time,end_time,dt),Sm/M,color='black',label='Susceptible Mosquitos')  
plt.plot(np.arange(start_time,end_time,dt),Im/M,color='magenta',label='Infected Mosquitos')  
plt.legend()
plt.xlabel('Time (in days)')
plt.ylabel('Fraction of mosquitos')
plt.title('Fraction of mosquitos of every category (Basic Model)')
plt.show()
```


![png](/pictures/malaria_model/output_3_0.png)



![png](/pictures/malaria_model/output_3_1.png)



```python
def sh(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return -delta*sh*im/N + alpha*ih

def ih(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return -beta*ih - gamma*ih - alpha*ih + delta*sh*im/M

def imh(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return beta*ih

def sm(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return kappa*M - delta*sm*ih/N - eta*sm - theta*sm

def im(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return delta*sm*ih/N - eta*im - theta*im
```


```python
start_time = 0
end_time = 1000
dt = 2
n = int((end_time - start_time)/dt)

Sh = np.zeros(n) 
Ih = np.zeros(n) 
Imh = np.zeros(n) 
Sm = np.zeros(n) 
Im = np.zeros(n) 

Sh[0] = 300
Ih[0] = 1
Imh[0] = 0
Sm[0] = 300
Im[0] = 0

alpha = 0.3
beta = 0.01
gamma = 0.005
kappa = 0.01
delta = 0.3
eta = 0.01
theta = 0.005

theta_values = [0.001,0.005,0.01]

N = Sh[0] + Ih[0] + Imh[0]
M = Sm[0] + Im[0]
for theta in theta_values :
  for i in range(1,len(Sh)):
    Sh[i] = Sh[i-1] + dt*sh(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Ih[i] = Ih[i-1] + dt*ih(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Imh[i] = Imh[i-1] + dt*imh(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Sm[i] = Sm[i-1] + dt*sm(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Im[i] = Im[i-1] + dt*im(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
  plt.plot(np.arange(start_time,end_time,dt),Sh/N,color='green',label='Susceptible Humans')  
  plt.plot(np.arange(start_time,end_time,dt),Ih/N,color='blue',label='Infected Humans')  
  plt.plot(np.arange(start_time,end_time,dt),Imh/N,color='red',label='Immune Humans')  
  plt.legend()
  plt.xlabel('Time (in days)')
  plt.ylabel('Fraction of humans')
  plt.title('Fraction of humans of every category (Fumigation rate = ' + str(theta) + ')')
  plt.show()
  plt.plot(np.arange(start_time,end_time,dt),Sm/M,color='black',label='Susceptible Mosquitos')  
  plt.plot(np.arange(start_time,end_time,dt),Im/M,color='magenta',label='Infected Mosquitos')  
  plt.legend()
  plt.xlabel('Time (in days)')
  plt.ylabel('Fraction of mosquitos')
  plt.title('Fraction of mosquitos of every category (Fumigation rate = ' + str(theta) + ')')
  plt.show()
```


![png](/pictures/malaria_model/output_5_0.png)



![png](/pictures/malaria_model/output_5_1.png)



![png](/pictures/malaria_model/output_5_2.png)



![png](/pictures/malaria_model/output_5_3.png)



![png](/pictures/malaria_model/output_5_4.png)



![png](/pictures/malaria_model/output_5_5.png)



```python
def sh(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return -delta*sh*im/N + alpha*ih - theta*sh

def ih(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return -beta*ih - gamma*ih - alpha*ih + delta*sh*im/M

def imh(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return beta*ih + theta*sh

def sm(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return kappa*M - delta*sm*ih/N - eta*sm

def im(sh,ih,imh,sm,im,alpha,beta,gamma,kappa,delta,eta,N,M,theta):
  return delta*sm*ih/N - eta*im
```


```python
start_time = 0
end_time = 1000
dt = 2
n = int((end_time - start_time)/dt)

Sh = np.zeros(n) 
Ih = np.zeros(n) 
Imh = np.zeros(n) 
Sm = np.zeros(n) 
Im = np.zeros(n) 

Sh[0] = 300
Ih[0] = 1
Imh[0] = 0
Sm[0] = 300
Im[0] = 0

alpha = 0.3
beta = 0.01
gamma = 0.005
kappa = 0.01
delta = 0.3
eta = 0.01
theta = 0.005

theta_values = [0.01,0.02,0.03,0.04]

N = Sh[0] + Ih[0] + Imh[0]
M = Sm[0] + Im[0]
for theta in theta_values :
  for i in range(1,len(Sh)):
    Sh[i] = Sh[i-1] + dt*sh(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Ih[i] = Ih[i-1] + dt*ih(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Imh[i] = Imh[i-1] + dt*imh(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Sm[i] = Sm[i-1] + dt*sm(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
    Im[i] = Im[i-1] + dt*im(Sh[i-1],Ih[i-1],Imh[i-1],Sm[i-1],Im[i-1],alpha,beta,gamma,kappa,delta,eta,N,M,theta)
  plt.plot(np.arange(start_time,end_time,dt),Sh/N,color='green',label='Susceptible Humans')  
  plt.plot(np.arange(start_time,end_time,dt),Ih/N,color='blue',label='Infected Humans')  
  plt.plot(np.arange(start_time,end_time,dt),Imh/N,color='red',label='Immune Humans')  
  plt.legend()
  plt.xlabel('Time (in days)')
  plt.ylabel('Fraction of humans')
  plt.title('Fraction of humans of every category (Vaccination rate = ' + str(theta) + ')')
  plt.show()
  plt.plot(np.arange(start_time,end_time,dt),Sm/M,color='black',label='Susceptible Mosquitos')  
  plt.plot(np.arange(start_time,end_time,dt),Im/M,color='magenta',label='Infected Mosquitos')  
  plt.legend()
  plt.xlabel('Time (in days)')
  plt.ylabel('Fraction of mosquitos')
  plt.title('Fraction of mosquitos of every category (Vaccination rate = ' + str(theta) + ')')
  plt.show()
```


![png](/pictures/malaria_model/output_7_0.png)



![png](/pictures/malaria_model/output_7_1.png)



![png](/pictures/malaria_model/output_7_2.png)



![png](/pictures/malaria_model/output_7_3.png)



![png](/pictures/malaria_model/output_7_4.png)



![png](/pictures/malaria_model/output_7_5.png)



![png](/pictures/malaria_model/output_7_6.png)



![png](/pictures/malaria_model/output_7_7.png)

