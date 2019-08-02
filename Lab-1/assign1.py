import random as rand


n = 1000000
p_act = 0.7
p1_act = 0.6
p2_act = 0.3

def DataGenerate():
	s = ""
	m = 0
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
	return s, m;

def expectancy(p, p1, p2, x):
	num = p * (p1**x) * ((1-p1)**(1-x))
	den = num + (1-p)*(p2**x)*((1-p2)**(1-x))
	return num/den

def calculate(p, p1, p2, data):
	sum1 = 0;
	sum2 = 0;
	for i in range(n):
		x = 0
		if(data[i] == 'H'):
			x = 1
		else:
			x = 0
		ezi = expectancy(p, p1, p2, x)
		sum1 += ezi
		sum2 += x*ezi
	return sum1, sum2

data, m = DataGenerate()
precision = 0.0000000000001
p = 0.7
p1 = 0.65
p2 = 0.1
p_pre = 0.0
p1_pre = 0.0
p2_pre = 0.0

times = 100
while((abs(p-p_pre) > precision or abs(p1-p1_pre) > precision or abs(p2-p2_pre) > precision) and times):
	times -= 1
	p_pre = p
	p1_pre = p1
	p2_pre = p2
	ex_sum, x_ex_sum = calculate(p, p1, p2, data)
	p = ex_sum/n
#	ex_sum, x_ex_sum = calculate(p, p1, p2, data)
	p1 = x_ex_sum/ex_sum
#	ex_sum, x_ex_sum = calculate(p, p1, p2, data)
	p2 = (m - x_ex_sum)/(n - ex_sum)
	print(p, p1, p2)
#if(abs(p-p_act) < precision and abs(p1-p1_act) < precision and abs(p2-p2_act) < precision):
