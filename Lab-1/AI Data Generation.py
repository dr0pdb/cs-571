import random as rand

n = 1000000000
p = 0.7
p1 = 0.6
p2 = 0.3


f = open("Data.txt", "w")
for i in range(n):
	a = rand.uniform(0, 1)
	if(a < p):
		b = rand.uniform(0, 1)
		if(b < p1):
			f.write("H")
		else:
			f.write("T")
	else:
		b = rand.uniform(0, 1)
		if(b < p2):
			f.write("H")
		else:
			f.write("T")
f.close()