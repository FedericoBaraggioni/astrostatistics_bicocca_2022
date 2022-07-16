import random
import numpy as np
from numba import jit

@jit
def Monty(N_doors,Doors,arr,prize,Win):
	player = random.randint(0,N_doors-1)
	presentatore = np.random.choice(np.delete(arr,[player,prize]),N_doors-2,replace=False)
	Cons=player
	switcher=np.random.choice(np.delete(arr,np.append(player,presentatore)))
	ext=np.random.choice(np.delete(arr,presentatore))

	if(Doors[Cons]==1):
		Win[0]+=1
	else:
		Win[1]+=1
	if(Doors[ext]==1):
		Win[2]+=1
	return Win


N_doors=3
N=10**5
arr = np.arange(N_doors)
Doors=np.zeros(N_doors)
prize=random.randint(0,N_doors-1)
Doors[prize]=1
Win=np.zeros(3)
for j in range(N):
	Win=Monty(N_doors,Doors,arr,prize,Win)


print("Conservative: ",100*Win[0]/N)
print("Switcher: ",100*Win[1]/N)
print("Ext: ",100*Win[2]/N)
