type(a1).
type(a2).
type(a3).
type(a4).
type(a5).
type(a6).

in(X1, a1).
in(X2, a1).
in(xor(X1, X2), a2).
in(C1, a2).
in(X1, a3).
in(X2, a3).
in(X1, a4).
in(X2, a4).
in(or(X1, X2), C1).
in(and(X1, X2), or(C1, and(X1,X2))).

in(a11, a1) :-
	connected(X1, in(1,a1)),
	connected(in(1,a1), X1),
	a11 .
in(a12, a1).
in(a21, a2).
in(a22, a2).
in(a31, a3).
in(a32, a3).
in(a41, a4).
in(a42, a4).
in(a51, a5).
in(a52, a5).
in(a61, a6).
in(a62, a6).

signal(a11).
signal(a12).
signal(a21).
signal(a22).
signal(a31).
signal(a32).
signal(a41).
signal(a42).
signal(a51).
signal(a52).
signal(a61).
signal(a62).

connected(X1, in(a11,a1)). 
connected(X2, in(a12, a1)).
connected(out(a1), in(a21, a2)).
connected(C1, in(a22, a2)).
connected(X1, in(a31,a3)).
connected(X2, in(a32, a3)).
connected(X1, in(a41,a4)). 
connected(X2, in(a42, a4)).
connected(out(a4), in(a51, a5)).
connected(C1, in(a52, a5)).
connected(out(a3), in(a61, a6)). 
connected(out(a5), in(a62, a6)).

connected(in(a11,a1), X1). 
connected(in(a12, a1), X2).
connected(in(a21, a2), out(a1)).
connected(in(a22, a2), C1).
connected(in(a31,a3), X1).
connected(in(a32, a3), X2).
connected(in(a41,a4), X1). 
connected(in(a42, a4), X2).
connected(in(a51, a5), out(a4)).
connected(in(a52, a5), C1).
connected(in(a61, a6), out(a3)). 
connected(in(a62, a6), out(a5)).

and(A,B) :- A,B.
or(A,B) :- A;B.
nand(A,B) :- not(and(A,B)).
nor(A,B) :- not(or(A,B)).
xor(A,B) :- or(A,B), nand(A,B).

out(a1) :- 
	connected(X1, in(a11,a1)),
	connected(X2, in(a12, a1)),
	out(a1) is xor(X1, X2).

out(a2) :- 
	cconnected(in(out(a1), a2), C2),
	xor(xor(X1, X2), C1),!.

out(a3) :- 
	connected(X1, in(,a3)),

	and(X2, X1).
out(a4) :- or(X2, X1).
out(a5) :- and(or(X2, X1), C1).
out(a6) :- or(and(X2, X1), and(or(X2, X1), C1)).

add(X1, X2, C1, Y, C2) :-
	Y is out(a1),
	C2 is out(a6).