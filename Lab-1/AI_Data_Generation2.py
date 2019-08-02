import random as rand


def expec(p, p1, p2, x):
	num = p * (p1**x) * ((1-p1)**(1-x))
	den = num + (1-p)*(p2**x)*((1-p2)**(1-x))
	return num/den

n = 1000
m = 0
p_act = 0.70
p1_act = 0.60
p2_act = 0.30
s = ""

for i in range(n):
	a = rand.uniform(0, 1)
	if(a < p_act):
		b = rand.uniform(0, 1)
		if(b < p1_act):
			s = s + "H"
			m += 1
		else:
			s = s + "T"
	else:
		b = rand.uniform(0, 1)
		if(b < p2_act):
			s = s + "H"
			m += 1
		else:
			s = s + "T"

precision = 0.01
for i1 in range(1, 10):
	for j in range(1, 10):
		for k in range(1, 10):
			p = i1/10
			p1 = j/10
			p2 = k/10
			p_pre = 0.0
			p1_pre = 0.0
			p2_pre = 0.0
			print("initial: ", p, p1, p2)
			times = 1000
			while((abs(p-p_pre) > precision or abs(p1-p1_pre) > precision or abs(p2-p2_pre) > precision) and times):
				times -= 1
				ppre = p
				p1pre = p1
				p2pre = p2
				ex_sum = 0.0
				x_ex_sum = 0.0
				for i in range(n):
					x = 0
					if(s[i] == 'H'):
						x = 1
					else:
						x = 0
					ezi = expec(p, p1, p2, x)
					ex_sum += ezi
					x_ex_sum += x*ezi
				p = ex_sum/n
				p1 = x_ex_sum/ex_sum
				p2 = (m - x_ex_sum)/(n - ex_sum)
			if(abs(p-p_act) < precision and abs(p1-p1_act) < precision and abs(p2-p2_act) < precision):
				print(p, p1, p2)
