xor(0,0,0).
xor(0,1,1).
xor(1,0,1).
xor(1,1,0).
or(0,0,0).
or(0,1,1).
or(1,0,1).
or(1,1,1).
and(0,0,0).
and(0,1,0).
and(1,0,0).
and(1,1,1).
not(0,1).
not(1,0).

halfa([X, Y], [Sum, Carry]) :-
xor(X, Y, Sum),
and(X,Y, Carry).

adder1([X,Y,CarryIn], [Sum, CarryOut]) :-
halfa([X,Y], [Sum1, Carry1]),
  halfa([Sum1,CarryIn], [Sum, Carry2]),
  or(Carry1, Carry2, CarryOut).

adder2([X1, X0], [Y1, Y0],[Sum1, Sum0, CarryOut]) :-
  adder1([X0, Y0, 0], [Sum0, Carry0]),
adder1([X1, Y1, Carry0], [Sum1, CarryOut]).

fadder2([X1, X0], [Y1, Y0], [CarryIn], [Sum1, Sum0, CarryOut]) :-
	adder1([X0, Y0, CarryIn], [Sum0, C1]), 
	adder1([X1, Y1, C1], [Sum1, CarryOut]).