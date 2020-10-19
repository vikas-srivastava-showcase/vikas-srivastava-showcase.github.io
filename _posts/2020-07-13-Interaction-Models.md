---
layout: post
title: "Interaction Model"
date: 2020-07-13
---

```python
import numpy as np
import matplotlib.pyplot as plt
import math
```


```python
start_time = 0
end_time = 5
dt = 0.01
n = int((end_time - start_time) / dt )

WTS = np.zeros((n,1))
BTS = np.zeros((n,1))
WTS[0] = 20
BTS[0] = 15
wts_birth = 1
bts_birth = 1
wts_death = 0.27
bts_death = 0.2

def model(birth,death,my_pop,other_pop):
  return my_pop*birth - death*other_pop*my_pop

for i in range(1,len(WTS)):
  WTS[i] = WTS[i-1] + model(wts_birth,wts_death,WTS[i-1],BTS[i-1]) * dt
  BTS[i] = BTS[i-1] + model(bts_birth,bts_death,BTS[i-1],WTS[i-1]) * dt
  
plt.plot(np.arange(start_time,end_time,dt),WTS,label='WTS')
plt.plot(np.arange(start_time,end_time,dt),BTS,label='BTS')
plt.xlabel('Time (in days)')
plt.title('Interaction model\nTwo species competing with each other')
plt.ylabel('Population')
plt.legend()
plt.show()
```


![png](/pictures/interaction_model/output_2_0.png)


We observe that BTS population increases after a certain point of time. Reason for this is that death proportionality constant for BTS population is lesser compared to that of WTS population death proportionality constant.


```python
start_time = 0
end_time = 25
dt = 0.01
n = int((end_time - start_time) / dt )

r1 = 0.4
K = 50
r2_values = [0.1,0.2,0.3,0.4,0.5,0.6]

def model(my_pop,r):
  return my_pop*r*(1 - my_pop / K)

for r2 in r2_values:
  WTS = np.zeros((n,1))
  BTS = np.zeros((n,1))
  WTS[0] = 20
  BTS[0] = 15
  for i in range(1,len(WTS)):
    WTS[i] = WTS[i-1] + model(WTS[i-1],r1) * dt
    BTS[i] = BTS[i-1] + model(BTS[i-1],r2) * dt

  plt.plot(np.arange(start_time,end_time,dt),WTS,label='WTS')
  plt.plot(np.arange(start_time,end_time,dt),BTS,label='BTS')
  plt.xlabel('Time (in days)')
  plt.title('No interaction model\nEach specie has a carrying capacity in the environment\nAnalysis when WTS initial population is greater than BTS')
  plt.ylabel('Population')
  plt.legend()
  plt.show()
```


![png](/pictures/interaction_model/output_4_0.png)



![png](/pictures/interaction_model/output_4_1.png)



![png](/pictures/interaction_model/output_4_2.png)



![png](/pictures/interaction_model/output_4_3.png)



![png](/pictures/interaction_model/output_4_4.png)



![png](/pictures/interaction_model/output_4_5.png)


The value of r1 and r2 determine the population of both species. As here there is no intercation between species and competition exists among species of one community only we see that r value determines how fast will the maximum population of constraint is reached. The more the value of r the more early saturation is reached in the population.


```python
start_time = 0
end_time = 15
dt = 0.01
n = int((end_time - start_time) / dt )

r1 = 0.4
K = 50
r2_values = [0.1,0.2,0.3,0.4,0.5,0.6]

def model(my_pop,r):
  return my_pop*r*(1 - my_pop / K)

for r2 in r2_values:
  WTS = np.zeros((n,1))
  BTS = np.zeros((n,1))
  WTS[0] = 15
  BTS[0] = 20
  for i in range(1,len(WTS)):
    WTS[i] = WTS[i-1] + model(WTS[i-1],r1) * dt
    BTS[i] = BTS[i-1] + model(BTS[i-1],r2) * dt

  plt.plot(np.arange(start_time,end_time,dt),WTS,label='WTS')
  plt.plot(np.arange(start_time,end_time,dt),BTS,label='BTS')
  plt.xlabel('Time (in days)')
  plt.title('No interaction model\nEach specie has a carrying capacity in the environment\nAnalysis when BTS initial population is greater than WTS')
  plt.ylabel('Population')
  plt.legend()
  plt.show()
```


![png](/pictures/interaction_model/output_6_0.png)



![png](/pictures/interaction_model/output_6_1.png)



![png](/pictures/interaction_model/output_6_2.png)



![png](/pictures/interaction_model/output_6_3.png)



![png](/pictures/interaction_model/output_6_4.png)



![png](/pictures/interaction_model/output_6_5.png)



```python
start_time = 0
end_time = 75
dt = 0.01
n = int((end_time - start_time) / dt )

r1 = 0.4
K = 50
r2_values = [0.1,0.2,0.3,0.4,0.5,0.6]

def model(my_pop,other_pop,r):
  return my_pop*r*(1 - (my_pop + other_pop) / K)

for r2 in r2_values:
  WTS = np.zeros((n,1))
  BTS = np.zeros((n,1))
  WTS[0] = 15
  BTS[0] = 20
  for i in range(1,len(WTS)):
    WTS[i] = WTS[i-1] + model(WTS[i-1],BTS[i-1],r1) * dt
    BTS[i] = BTS[i-1] + model(BTS[i-1],WTS[i-1],r2) * dt

  plt.plot(np.arange(start_time,end_time,dt),WTS,label='WTS')
  plt.plot(np.arange(start_time,end_time,dt),BTS,label='BTS')
  plt.plot(np.arange(start_time,end_time,dt),WTS+BTS,label='BTS + WTS')
  plt.xlabel('Time (in days)')
  plt.title('Competition model\nWhen two species fight for same resource\nAnalysis when BTS initial population is greater than WTS initial popultion')
  plt.ylabel('Population')
  plt.legend()
  plt.show()


```


![png](/pictures/interaction_model/output_7_0.png)



![png](/pictures/interaction_model/output_7_1.png)



![png](/pictures/interaction_model/output_7_2.png)



![png](/pictures/interaction_model/output_7_3.png)



![png](/pictures/interaction_model/output_7_4.png)



![png](/pictures/interaction_model/output_7_5.png)



```python
start_time = 0
end_time = 15
dt = 0.01
n = int((end_time - start_time) / dt )

r1 = 0.4
K = 50
r2_values = [0.1,0.2,0.3,0.4,0.5,0.6]

def model(my_pop,other_pop,r):
  return my_pop*r*(1 - (my_pop + other_pop) / K)

for r2 in r2_values:
  WTS = np.zeros((n,1))
  BTS = np.zeros((n,1))
  WTS[0] = 20
  BTS[0] = 15
  for i in range(1,len(WTS)):
    WTS[i] = WTS[i-1] + model(WTS[i-1],BTS[i-1],r1) * dt
    BTS[i] = BTS[i-1] + model(BTS[i-1],WTS[i-1],r2) * dt

  plt.plot(np.arange(start_time,end_time,dt),WTS,label='WTS')
  plt.plot(np.arange(start_time,end_time,dt),BTS,label='BTS')
  plt.plot(np.arange(start_time,end_time,dt),WTS+BTS,label='BTS + WTS')
  plt.xlabel('Time (in days)')
  plt.title('Competition model\nWhen two species fight for same resource\nAnalysis when WTS initial population is greater than BTS initial popultion')
  plt.ylabel('Population')
  plt.legend()
  plt.show()
```


![png](/pictures/interaction_model/output_8_0.png)



![png](/pictures/interaction_model/output_8_1.png)



![png](/pictures/interaction_model/output_8_2.png)



![png](/pictures/interaction_model/output_8_3.png)



![png](output_8_4.png)



![png](/pictures/interaction_model/output_8_5.png)



```python
start_time = 0
end_time = 15
dt = 0.01
n = int((end_time - start_time) / dt )

hawks = np.zeros((n,1))
squirrel = np.zeros((n,1))
hawks[0] = 15
squirrel[0] = 100

def model_predator(birth,death,predator_pop,prey_pop):
  return birth*(predator_pop*prey_pop) - death*predator_pop

def model_prey(birth,death,predator_pop,prey_pop):
  return birth*prey_pop - death*(prey_pop*predator_pop)

hawks_birth = 0.01
hawks_death = 1.06
squirrel_birth = 2
squirrel_death = 0.02

for i in range(1,len(hawks)):
  hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1]) * dt
  squirrel[i] = squirrel[i-1] + model_prey(squirrel_birth,squirrel_death,hawks[i-1],squirrel[i-1]) * dt

plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
plt.title('Prey Predator model')
plt.xlabel('Time (in days)')
plt.ylabel('Population')
plt.legend()
plt.show()

plt.plot(squirrel,hawks)
plt.xlabel('Squirrel')
plt.title('Prey Predator Model')
plt.ylabel('Hawks')
plt.show()
```


![png](/pictures/interaction_model/output_9_0.png)



![png](/pictures/interaction_model/output_9_1.png)


Shows 3 cycles have been done till time t=15


```python
start_time = 0
end_time = 10
dt = 0.01
n = int((end_time - start_time) / dt )

def model_predator(birth,death,predator_pop,prey_pop):
  return birth*(predator_pop*prey_pop) - death*predator_pop

def model_prey(birth,death,predator_pop,prey_pop):
  return birth*prey_pop - death*(prey_pop*predator_pop)

init_prey = [100,125,150,175]
init_predator = [15,75,135,195]

for i in range(len(init_prey)):

  hawks = np.zeros((n,1))
  squirrel = np.zeros((n,1))
  hawks[0] = init_predator[i]
  squirrel[0] = init_prey[i]

  hawks_birth = 0.01
  hawks_death = 1.06
  squirrel_birth = 2
  squirrel_death = 0.02

  for i in range(1,len(hawks)):
    hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1]) * dt
    squirrel[i] = squirrel[i-1] + model_prey(squirrel_birth,squirrel_death,hawks[i-1],squirrel[i-1]) * dt

#   plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
#   plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
#   plt.xlabel('Time (in days)')
#   plt.ylabel('Population')
#   plt.legend()
#   plt.show()

  plt.plot(squirrel,hawks,label='Hawks_init = ' + str(hawks[0][0]) + ' Squirrel_init = ' + str(squirrel[0][0]))
  plt.xlabel('Squirrel')
  plt.ylabel('Hawks')
#   plt.show()

plt.title('Prey Predator model\nAnalysis for different initial population of prey and predator')
plt.legend()
plt.show()
```


![png](/pictures/interaction_model/output_11_0.png)



```python
start_time = 0
end_time = 15
dt = 0.01
n = int((end_time - start_time) / dt )

hawks = np.zeros((n,1))
squirrel = np.zeros((n,1))
hawks[0] = 100
squirrel[0] = 106

def model_predator(birth,death,predator_pop,prey_pop):
  return birth*(predator_pop*prey_pop) - death*predator_pop

def model_prey(birth,death,predator_pop,prey_pop):
  return birth*prey_pop - death*(prey_pop*predator_pop)

hawks_birth = 0.01
hawks_death = 1.06
squirrel_birth = 2
squirrel_death = 0.02

for i in range(1,len(hawks)):
  hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1]) * dt
  squirrel[i] = squirrel[i-1] + model_prey(squirrel_birth,squirrel_death,hawks[i-1],squirrel[i-1]) * dt

plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
plt.xlabel('Time (in days)')
plt.title('Prey Predator Model\nEquilibrium situation')
plt.ylabel('Population')
plt.legend()
plt.show()
```


![png](/pictures/interaction_model/output_12_0.png)



```python
start_time = 0
end_time = 5
dt = 1
n = int((end_time - start_time) / dt )

hawks = np.zeros((n,1))
squirrel = np.zeros((n,1))
hawks[0] = 500
squirrel[0] = 3000 

def model_predator(birth,death,predator_pop,prey_pop):
  return birth*(predator_pop*prey_pop) - death*predator_pop

def model_prey(birth,death,predator_pop,prey_pop):
  return birth*prey_pop - death*(prey_pop*predator_pop)

hawks_death = 3000*0.1
hawks_birth = 0.1
squirrel_death = 2/500
squirrel_birth = 2

for i in range(1,len(hawks)):
  hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1]) * dt
  squirrel[i] = squirrel[i-1] + model_prey(squirrel_birth,squirrel_death,hawks[i-1],squirrel[i-1]) * dt

plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
plt.xlabel('Time (in days)')
plt.ylabel('Population')
plt.title('Prey Predator model\nEquilibrium situation')
plt.legend()
plt.show()
```


![png](/pictures/interaction_model/output_13_0.png)



```python
start_time = 0
end_time = 10
dt = 0.01
n = int((end_time - start_time) / dt )

hawks = np.zeros((n,1))
squirrel = np.zeros((n,1))
hawks[0] = 15
squirrel[0] = 100

def model_predator(birth,death,predator_pop,prey_pop,M):
  return birth*(predator_pop*prey_pop) - (birth*(predator_pop**2)*prey_pop/M)

def model_prey(birth,death,predator_pop,prey_pop):
  return birth*prey_pop - death*(prey_pop*predator_pop)

hawks_birth = 0.01
hawks_death = 1.06
squirrel_birth = 2
squirrel_death = 0.02
M = 300

for i in range(1,len(hawks)):
  hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1],M) * dt
  squirrel[i] = squirrel[i-1] + model_prey(squirrel_birth,squirrel_death,hawks[i-1],squirrel[i-1]) * dt

plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
plt.xlabel('Time (in days)')
plt.ylabel('Population')
plt.legend()
plt.show()

plt.plot(squirrel,hawks)
plt.xlabel('Squirrel')
plt.ylabel('Hawks')
plt.show()
```


![png](/pictures/interaction_model/output_14_0.png)



![png](/pictures/interaction_model/output_14_1.png)



```python
start_time = 0
end_time = 50
dt = 0.01
n = int((end_time - start_time) / dt )

hawks = np.zeros((n,1))
squirrel = np.zeros((n,1))
hawks[0] = 15
squirrel[0] = 100
time = np.arange(start_time,end_time,dt)


def model_predator(birth,death,predator_pop,prey_pop):
  return birth*(predator_pop*prey_pop) - death*predator_pop

def model_prey(f,a,p,time,death,predator_pop,prey_pop):
  return (f+a*math.cos(p*time))*prey_pop - death*(prey_pop*predator_pop)

hawks_birth = 0.01
hawks_death = 1.06
squirrel_death = 0.02
f = 2
a = 1
p = 2

for i in range(1,len(hawks)):
  hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1]) * dt
  squirrel[i] = squirrel[i-1] + model_prey(f,a,p,time[i-1],squirrel_death,hawks[i-1],squirrel[i-1]) * dt

plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
plt.xlabel('Time (in days)')
plt.ylabel('Population')
plt.legend()
plt.show()

plt.plot(squirrel,hawks)
plt.xlabel('Squirrel')
plt.ylabel('Hawks')
plt.show()

for i in range(len(squirrel)):
  if(time[i]%10==0):
    print(round(time[i],3) , round(squirrel[i][0],3) , round(hawks[i][0],3))
```


![png](/pictures/interaction_model/output_15_0.png)



![png](/pictures/interaction_model/output_15_1.png)


    0.0 100.0 15.0
    10.0 0.283 23.775
    20.0 72.548 5.742
    30.0 313.061 1.641
    40.0 664.193 7.974



```python
start_time = 0
end_time = 50
dt = 0.01
n = int((end_time - start_time) / dt )

hawks = np.zeros((n,1))
squirrel = np.zeros((n,1))
hawks[0] = 15
squirrel[0] = 100

def model_predator(birth,death,predator_pop,prey_pop,human_hunt):
  return birth*(predator_pop*prey_pop) - death*predator_pop - human_hunt*predator_pop

def model_prey(birth,death,predator_pop,prey_pop,human_hunt):
  return birth*prey_pop - death*(prey_pop*predator_pop) - human_hunt*prey_pop

hawks_birth = 0.01
hawks_death = 1.06
squirrel_birth = 2
squirrel_death = 0.02
human_hunts = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1]

for human_hunt in human_hunts:

  for i in range(1,len(hawks)):
    hawks[i] = hawks[i-1] + model_predator(hawks_birth,hawks_death,hawks[i-1],squirrel[i-1],human_hunt) * dt
    squirrel[i] = squirrel[i-1] + model_prey(squirrel_birth,squirrel_death,hawks[i-1],squirrel[i-1],human_hunt) * dt

  plt.plot(np.arange(start_time,end_time,dt),hawks,label='Predator')
  plt.plot(np.arange(start_time,end_time,dt),squirrel,label='Prey')
  plt.xlabel('Time (in days)')
  plt.ylabel('Population')
  plt.title('Human_Hunt rate : ' + str(human_hunt))
  plt.legend()
  plt.show()

  plt.plot(squirrel,hawks)
  plt.xlabel('Squirrel')
  plt.title('Human_Hunt rate : ' + str(human_hunt))
  plt.ylabel('Hawks')
  plt.show()
```


![png](/pictures/interaction_model/output_16_0.png)



![png](/pictures/interaction_model/output_16_1.png)



![png](/pictures/interaction_model/output_16_2.png)



![png](/pictures/interaction_model/output_16_3.png)



![png](/pictures/interaction_model/output_16_4.png)



![png](/pictures/interaction_model/output_16_5.png)



![png](/pictures/interaction_model/output_16_6.png)



![png](/pictures/interaction_model/output_16_7.png)



![png](/pictures/interaction_model/output_16_8.png)



![png](/pictures/interaction_model/output_16_9.png)



![png](/pictures/interaction_model/output_16_10.png)



![png](/pictures/interaction_model/output_16_11.png)



![png](/pictures/interaction_model/output_16_12.png)



![png](/pictures/interaction_model/output_16_13.png)



![png](/pictures/interaction_model/output_16_14.png)



![png](/pictures/interaction_model/output_16_15.png)



![png](/pictures/interaction_model/output_16_16.png)



![png](/pictures/interaction_model/output_16_17.png)



![png](/pictures/interaction_model/output_16_18.png)



![png](/pictures/interaction_model/output_16_19.png)



![png](/pictures/interaction_model/output_16_20.png)



![png](/pictures/interaction_model/output_16_21.png)



![png](/pictures/interaction_model/output_16_22.png)



![png](/pictures/interaction_model/output_16_23.png)



![png](/pictures/interaction_model/output_16_24.png)



![png](/pictures/interaction_model/output_16_25.png)



![png](/pictures/interaction_model/output_16_26.png)



![png](/pictures/interaction_model/output_16_27.png)



![png](/pictures/interaction_model/output_16_28.png)



![png](/pictures/interaction_model/output_16_29.png)



![png](/pictures/interaction_model/output_16_30.png)



![png](/pictures/interaction_model/output_16_31.png)



![png](/pictures/interaction_model/output_16_32.png)



![png](/pictures/interaction_model/output_16_33.png)



![png](/pictures/interaction_model/output_16_34.png)



![png](/pictures/interaction_model/output_16_35.png)



![png](/pictures/interaction_model/output_16_36.png)



![png](/pictures/interaction_model/output_16_37.png)



![png](/pictures/interaction_model/output_16_38.png)



![png](/pictures/interaction_model/output_16_39.png)



![png](/pictures/interaction_model/output_16_40.png)



![png](/pictures/interaction_model/output_16_41.png)

