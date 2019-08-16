belong(a).
belong(b).
belong(c). 
belong(X) :- 
	\+mc(X), \+sk(X), !, fail. 
belong(X).
like(a,rain).
like(a,snow).
like(a,X) :- 
	\+ like(b, X).
like(b,X) :- 
	like(a,X), !, fail.
like(b,X).
mc(X) :- 
	like(X, rain), !, fail.
mc(X).
sk(X) :- 
	\+like(X, snow), !, fail.
sk(X).
g(X) :- belong(X), mc(X), \+sk(X), !.