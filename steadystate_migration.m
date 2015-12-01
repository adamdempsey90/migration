#!/Applications/Mathematica.app/Contents/MacOS/WolframScript -script


PlanetFlag = ToExpression[$ScriptCommandLine[[2]]];
Print["PlanetFlag = ", PlanetFlag]

a0 = ToExpression[$ScriptCommandLine[[3]]];
Print["a0 = ", a0]
\[Gamma] = ToExpression[$ScriptCommandLine[[4]]];
Print["\[Gamma] = ", \[Gamma]]
\[Mu] = ToExpression[$ScriptCommandLine[[5]]];
Print["\[Mu] = ",\[Mu]]

h = ToExpression[$ScriptCommandLine[[6]]];
Print["h  = ",  h]
\[Alpha] = ToExpression[$ScriptCommandLine[[7]]];
Print["\[Alpha] = ", \[Alpha]]

\[Beta] = ToExpression[$ScriptCommandLine[[8]]];
Print["\[Beta] = ", \[Beta]]

\[CapitalDelta] = ToExpression[$ScriptCommandLine[[9]]];
Print["\[CapitalDelta] = ",\[CapitalDelta]]

\[CapitalGamma] = ToExpression[$ScriptCommandLine[[10]]];
Print["\[CapitalGamma] = ",\[CapitalGamma]] 

ri = ToExpression[$ScriptCommandLine[[11]]];
Print["ri = ", ri]

ro = ToExpression[$ScriptCommandLine[[12]]];
Print["ro = ", ro]

\[Lambda]i = ToExpression[$ScriptCommandLine[[13]]];
Print["\[Lambda]i  = ", \[Lambda]i]

\[Lambda]o = ToExpression[$ScriptCommandLine[[14]]];
Print["\[Lambda]o = ", \[Lambda]o]

\[CapitalLambda]f = ToExpression[$ScriptCommandLine[[15]]];
Print["\[CapitalLambda]f = ",\[CapitalLambda]f]

\[CapitalLambda]c = ToExpression[$ScriptCommandLine[[16]]];
Print["\[CapitalLambda]c = ", \[CapitalLambda]c]

\[CapitalLambda]w= ToExpression[$ScriptCommandLine[[17]]];
Print["\[CapitalLambda]w  = ", \[CapitalLambda]w]

UseGaussian = ToExpression[$ScriptCommandLine[[18]]]
Print["UseGaussian = ", UseGaussian]

nr = ToExpression[$ScriptCommandLine[[19]]];
Print["nr = ", nr]

OneSided = ToExpression[$ScriptCommandLine[[20]]];

ai = ToExpression[$ScriptCommandLine[[21]]];
Print["a low = ", ai]
ao = ToExpression[$ScriptCommandLine[[22]]];
Print["a hi = ", ao]
Na = ToExpression[$ScriptCommandLine[[23]]];
Print["Number of a's = ", Na]

UseTorque = Mod[1+UseGaussian,2]
Print["UseTorque =", UseTorque]


T0[a_] := 2 Pi a \[Mu]^2 h^3/h;


rh[a_] := (\[Mu] h^3/3)^(1./3) a;


q[a_] := Max[h,rh[a]];




\[Alpha]disk[x_,rout_]:=\[Lambda]i + (\[Lambda]o-\[Lambda]i) (x/nu[x] - ri/nu[ri])/(rout/nu[rout]-ri/nu[ri]) 
\[Alpha]disk2[x_,lamin_,lamout_,rin_,rout_]:=lamin+ (lamout-lamin) (x/nu[x] - rin/nu[rin])/(rout/nu[rout]-rin/nu[rin])
nu[x_]:= \[Alpha] h^2 x^\[Gamma]
vrn[x_,rout_]:= -1.5 (\[Lambda]o-\[Lambda]i)/(rout/nu[rout]-ri/nu[ri]) /\[Alpha]disk[x,rout]
\[Xi][x_,a_]:= (x-a)/q[a]
Func[x_,a_]:= T0 *((\[CapitalGamma]+1)Exp[-((\[Xi][x,a]-\[Beta])/\[CapitalDelta])^2] - (1-OneSided) Exp[-((\[Xi][x,a]+\[Beta])/\[CapitalDelta])^2])/(\[CapitalDelta] Sqrt[Pi])
s[x_,x0_,w_]:= .5(1 + Tanh[(x-x0)/w])
\[CapitalLambda]L[x_,a_,f_,c_]:= Piecewise[{{-f a Pi (\[Mu] h^3)^2 (x/(Max[h x, Abs[x-a]]))^4,x<a},{0,x>=a}}]
\[CapitalLambda]R[x_,a_,f_,c_]:= Piecewise[{{f a  Pi (\[Mu] h^3)^2 (a/(Max[h x, Abs[x-a]]))^4,x>a} ,{0,x<=a}}]
\[CapitalLambda][x_,a_,f_,c_]:=Piecewise[{{- f Pi (\[Mu] h^3)^2 (x/(x-a))^4,x<(a-c rh[a])},{0,a-c rh[a] <= x<= a+c rh[a]},{f Pi (\[Mu] h^3)^2 (x/(x-a))^4,x>(a+c rh[a])}}]
\[CapitalLambda]smooth[x_,a_,f_,c_,w_]:=(1-s[x,a-c h x,w])\[CapitalLambda]L[x,a,f,c] (1-OneSided)  + s[x,a-c h x,w] s[x,a + c h x,w] \[CapitalLambda]R[x,a,f,c]
Torque[r_,a_,f_,c_,w_]:= UseGaussian Func[r,a] + UseTorque \[CapitalLambda]smooth[r,a,f,c,w]
Mdot[a_]:= m[r] /. NDSolve[ {3 nu[r] \[Lambda]'[r] + \[Lambda][r](3 nu[r](\[Gamma]-.5)/r - PlanetFlag  Torque[r,a,\[CapitalLambda]f,\[CapitalLambda]c,\[CapitalLambda]w] /(Pi  Sqrt[r])) == m[r],m'[r]== 0,\[Lambda][ri]== \[Lambda]i,\[Lambda][ro]== \[Lambda]o},{\[Lambda],m},{r,ri,ro}];
sol[a_] :=  NDSolve[ {3 nu[r] \[Lambda]'[r] + \[Lambda][r](3 nu[r](\[Gamma]-.5)/r - PlanetFlag  Torque[r,a,\[CapitalLambda]f,\[CapitalLambda]c,\[CapitalLambda]w] /(Pi  Sqrt[r])) == m[r],m'[r]== 0,\[Lambda][ri]== \[Lambda]i,\[Lambda][ro]== \[Lambda]o},{\[Lambda],m},{r,ri,ro}]
adot[a_,mdot_]:= -Sqrt[a]/(\[Mu] h^3) ( 3 \[Lambda]o nu[ro]/Sqrt[ro] - 3 \[Lambda]i nu[ri]/Sqrt[ri] - 2 mdot (Sqrt[ro]-Sqrt[ri]))
asol[t_] := y[t] /. NDSolve[{y'[x] == adot[y[x]],y[0]== a0} ,{x,0,tend}]



avals = Table[ai + 1. (i-1) (ao-ai)/(Na-1) ,{i,1,Na}]
Print[avals]
rvals = Table[ ri + 1. (i-1) (ro-ri)/(nr-1),{i,1,nr}]

sols = Table[sol[avals[[i]]],{i,1,Na}]
mdotfinal = Table[m[avals[[i]]]/. sols[[i]][[1]],{i,1,Na}]
lamfinal = Table[\[Lambda][r] /. sols[[i]][[1]],{i,1,Na}]
adotfinal = Table[adot[avals[[i]],mdotfinal[[i]]],{i,1,Na}]
Print[mdotfinal]
Print[adotfinal] 
lamp= Table[\[Lambda][avals[[i]]] /. sols[[i]][[1]],{i,1,Na}]
Print[lamp]



(*mdotfinal = m[ri] /. sol*)

(*dTr = Table[Torque[rvals[[i]],a,\[CapitalLambda]f,\[CapitalLambda]c,\[CapitalLambda]w],{i,1,nr}]*)


(*dat = Table[{rvals[[i]],lamfinal[[i]][[1]],-mdotfinal[[1]]/(lamfinal[[i]][[1]]),dTr[[i]]},{i,1,nr}]*)
dat=Table[Join[{avals[[i]]},{adotfinal[[i]]},{mdotfinal[[i]]},{lamp[[i]]},Table[\[Lambda][rvals[[j]]] /. sols[[i]][[1]],{j,1,nr}],Table[Torque[rvals[[j]],avals[[i]],\[CapitalLambda]f,\[CapitalLambda]c,\[CapitalLambda]w],{j,1,nr}]],{i,1,Na}]
dat = AppendTo[dat,Join[{0,0,0,0},rvals,rvals]]

Export["results.dat",dat]

Print["Done"]

Exit[]


