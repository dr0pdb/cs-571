signal(X) :-
	X is X2, !.

add(X1, X2) :-
	signal(X1),
	signal(X2), !.