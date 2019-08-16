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


connected(X1, in(X1,a1)). 
connected(X2, in(X2, a1)).
connected(C1, in(out(a1), a2)).
connected(X1, in(X1,a3)).
connected(X2, in(X2, a3)).
connected(X1, in(1,a4)). 
connected(X2, in(2, a4)).
connected(C1, in(2, a5)).
connected(out(a1), in(1, a2)).
connected(out(a4), in(1, a5)). 
connected(out(a3), in(1, a6)). 
connected(out(a5), in(2, a6)).

and(A,B) :- A,B.
or(A,B) :- A;B.
nand(A,B) :- not(and(A,B)).
nor(A,B) :- not(or(A,B)).
xor(A,B) :- or(A,B), nand(A,B).

out(a1) :- 
	connected(X1, in(X1,a1)),
	connected(X2, in(X2, a1)),
	xor(X1, X2),!.

out(a2) :- 
	connected(C1, in(out(a1), a2)),
	connected(out(a1), in(1, a2)),
	xor(xor(X1, X2), C1),!.

out(a3) :- 
	connected(X1, in(,a3)),

	and(X2, X1).
out(a4) :- or(X2, X1).
out(a5) :- and(or(X2, X1), C1).
out(a6) :- or(and(X2, X1), and(or(X2, X1), C1)).

add(X1, X2, C1) :-
	format('x1 = ~w  x2 = ~w  x3 = ~w  ~n', [X1, X2, C1]),
	out(a1),
	out(a2),
	out(a3),
	out(a4),
	out(a5),
	out(a6).