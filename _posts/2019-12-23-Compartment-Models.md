

```python
import numpy as np
import math
import matplotlib.pyplot as plt
```


```python
w = (325*2)
volume = 3000

q0 = w/volume
t_half = 3.2
k = math.log(2)/t_half

start_time = 0
end_time = 10
dt = 0.01

t = np.arange(start_time,end_time,dt)

def model_a(k,q):
   return -k*q

n = int((end_time - start_time) / dt )
q = np.zeros(n)
q[0] = q0

for i in range(1,len(q)):
  q[i] = q[i-1] + (dt * model_a(k,q[i-1]))
  
plt.plot(t,q)
plt.ylabel('Concentration in microgram/mL')
plt.xlabel('Time in hours')
plt.title('One Compartment Model\nDrug Concentration in GI tract (only one compartment)')
plt.show()
```


![png](/pictures/comp_models/output_2_0.png)



```python
def model_b_x(k,x):
  return -k*x

def model_b_y(k1,k2,x,y):
  return k1*x - k2*y

w = (325*2)
volume = 3000

q0 = w/volume
k1 = 1.386
k2 = 0.1386

start_time = 0
end_time = 10
dt = 0.01

t = np.arange(start_time,end_time,dt)

n = int((end_time - start_time) / dt )
y = np.zeros(n)
x = np.zeros(n)

y[0] = 0
x[0] = q0

for i in range(1,len(q)):
  x[i] = x[i-1] + (dt * model_b_x(k1,x[i-1]))
  y[i] = y[i-1] + (dt * model_b_y(k1,k2,x[i],y[i-1]))
  
plt.plot(t,x)
plt.ylabel('Concentration in microgram/mL')
plt.xlabel('Time in hours')
plt.title('Two Compartment model\nDrug Concentration in Compartment 1 (GI Tract)')
plt.show()
plt.plot(t,y)
plt.ylabel('Concentration in microgram/mL')
plt.xlabel('Time in hours')
plt.title('Two Compartment model\nDrug Concentration in Compartment 2 (blood stream)')
plt.show()
print('The maximum concentration of drug in blood stream is ' , str(round(np.max(y),2)) , 'microgram/mL')
```


![png](/pictures/comp_models/output_3_0.png)



![png](/pictures/comp_models/output_3_1.png)


    The maximum concentration of drug in blood stream is  0.17 microgram/mL



```python
def model_b_x(k,x):
  return -k*x

def model_b_y(k1,k2,x,y):
  return k1*x - k2*y


w = (325*2)
volume = 3000

q0 = w/volume
k1 = 1.386

start_time = 0
end_time = 10
dt = 0.01

t = np.arange(start_time,end_time,dt)

n = int((end_time - start_time) / dt )

k2_array = [0.01386,0.06,0.1386,0.6386,1.386]
for k2 in k2_array:

  y = np.zeros(n)
  x = np.zeros(n)

  y[0] = 0
  x[0] = q0

  for i in range(1,len(q)):
    x[i] = x[i-1] + (dt * model_b_x(k1,x[i-1]))
    y[i] = y[i-1] + (dt * model_b_y(k1,k2,x[i],y[i-1]))

  string = 'k2 = ' + str(k2)
  plt.plot(t,y,label = string)
  plt.ylabel('Concentration in microgram/mL')
  plt.xlabel('Time in hours')

plt.title('Two compartment model\nDrug concentration in compartment 2 (blood stream)\nAnalysis for different rates of elimination from blood stream')
plt.legend(loc='upper right')
plt.show()
```


![png](/pictures/comp_models/output_4_0.png)



```python
def model_b_x(k,x):
  return -k*x

def model_b_y(k1,k2,x,y):
  return k1*x - k2*y

w = (325*2)
volume = 3000

q0 = w/volume
k2 = 0.0231

start_time = 0
end_time = 10
dt = 0.01

t = np.arange(start_time,end_time,dt)

n = int((end_time - start_time) / dt )

k1_array = [0.06931,0.11,0.691,1.0,1.5]

for k1 in k1_array:

  y = np.zeros(n)
  x = np.zeros(n)

  y[0] = 0
  x[0] = q0

  for i in range(1,len(q)):
    x[i] = x[i-1] + (dt * model_b_x(k1,x[i-1]))
    y[i] = y[i-1] + (dt * model_b_y(k1,k2,x[i],y[i-1]))

  string = 'k1 = ' + str(k1)
  plt.plot(t,y,label = string)
  plt.ylabel('Concentration in microgram/mL')
  plt.xlabel('Time in hours')
plt.title('Two compartment model\nDrug concentration in compartment 2 (blood stream)\nAnalysis for different rates of elimination from GI tract')
plt.legend(loc='upper right')
plt.show()
```


![png](/pictures/comp_models/output_5_0.png)



```python
def model_b_x(k,x):
  return -k*x

def model_b_y(k1,k2,x,y):
  return k1*x - k2*y

w = (325*2)
volume = 3000

q0 = w/volume
k2 = 0.0231

start_time = 0
end_time = 10
dt = 0.01

t = np.arange(start_time,end_time,dt)

n = int((end_time - start_time) / dt )

k1_array = [0.06931,0.11,0.691,1.0,1.5]

for k1 in k1_array:

  y = np.zeros(n)
  x = np.zeros(n)

  y[0] = 0
  x[0] = q0

  for i in range(1,len(q)):
    x[i] = x[i-1] + (dt * model_b_x(k1,x[i-1]))
    y[i] = y[i-1] + (dt * model_b_y(k1,k2,x[i],y[i-1]))

  string = 'k1 = ' + str(k1)
  plt.plot(t,x,label = string)
  plt.ylabel('Concentration in microgram/mL')
  plt.xlabel('Time in hours')
plt.title('Two compartment model\nDrug concentration in compartment 1 (GI tract)\nAnalysis for different rates of elimination from GI tract')
plt.legend(loc='upper right')
plt.show()
```


![png](/pictures/comp_models/output_6_0.png)



```python
def model_b_x(k,x):
  return -k*x

def model_b_y(k1,k2,x,y):
  return k1*x - k2*y

w = (325*2)
volume = 3000

q0 = w/volume
k1 = 1.386
k2 = 0.1386


start_time = 0
end_time = 24
dt = 0.001

t = np.arange(start_time,end_time,dt)
n = int((end_time - start_time) / dt )

y = np.zeros(n)
x = np.zeros(n)

y[0] = 0
x[0] = q0

for i in range(1,len(y)):
  if(t[i]%8 == 2):
    x[i] = x[i-1] + x[0] + (dt * model_b_x(k1,x[i-1]))
  else:
    x[i] = x[i-1] + (dt * model_b_x(k1,x[i-1]))
  y[i] = y[i-1] + (dt * model_b_y(k1,k2,x[i],y[i-1]))

plt.plot(t,y)
plt.ylabel('Concentration in microgram/mL')
plt.xlabel('Time in hours')
plt.title('Two compartment model\nDrug concentration in compartment 2 (blood stream)\nIntake of drug at regular time intervals (8 hours)')
plt.show()

plt.plot(t,x)
plt.ylabel('Concentration in microgram/mL')
plt.xlabel('Time in hours')
plt.title('Two compartment model\nDrug concentration in compartment 1 (GI Tract)\nIntake of drug at regular time intervals (8 hours)')
plt.show()
```


![png](/pictures/comp_models/output_7_0.png)



![png](/pictures/comp_models/output_7_1.png)



```python
def model_b_x(k,x):
  return -k*x

def model_b_y(k1,k2,x,y):
  return k1*x - k2*y

w = (325*3)
volume = 3000

q0 = w/volume

k1 =  0.6931  
k2 =  0.0231

start_time = 0
end_time = 200
dt = 1/2

t = np.arange(start_time,end_time,dt)
n = int((end_time - start_time) / dt )

y = np.zeros(n)
x = np.zeros(n)
    
y[0] = 0
x[0] = q0

for i in range(1,len(y)):
  j = i%12
  if(j==0):
    x[i] = q0
    
  x[i] = x[i] + x[i-1] + (dt * model_b_x(k1,x[i-1]))
  y[i] = y[i-1] + (dt * model_b_y(k1,k2,x[i],y[i-1]))
  
plt.plot(t,y)
plt.ylabel('Concentration in bloodstream in microgram/mL')
plt.xlabel('Time in hours')
plt.title('Two compartment model\nDrug concentration in compartment 2 (blood stream)\nIntake of drug at regular time intervals 6 pills at every 1/2 hour after 6 hours')
plt.show()

plt.plot(t,x)
plt.ylabel('Concentration in GI Tract in microgram/mL')
plt.xlabel('Time in hours')
plt.title('Two compartment model\nDrug concentration in compartment 2 (GI Tract)\nIntake of drug at regular time intervals 6 pills at every 1/2 hour after 6 hours')
plt.show()

for i in range(len(y)):
  if(int(y[i]*3000/325) == 20):
    print('The drug has its adverse effect after' , int(t[i]) , 'hours')
    break
```


![png](/pictures/comp_models/output_8_0.png)



![png](/pictures/comp_models/output_8_1.png)


    The drug has its adverse effect after 103 hours



```python

```
