#!/Applications/Mathematica.app/Contents/MacOS/WolframScript -script


PlanetFlag = ToExpression[$ScriptCommandLine[[2]]];
Print["PlanetFlag = ", PlanetFlag]
a= ToExpression[$ScriptCommandLine[[3]]];
Print["a = ", a]
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


UseTorque = Mod[1+UseGaussian,2]
Print["UseTorque =", UseTorque]


T0 = 2 Pi a \[Mu]^2 h^3/h;
Print[Evaluate[ri+ro]]

rh = (\[Mu] h^3/3)^(1./3) a;
Print["rh =" ,rh]

q = Max[h,rh];
Print["q = " ,q]



\[Alpha]disk[x_, 
  rout_] := \[Lambda]i + (\[Lambda]o - \[Lambda]i) (x/nu[x] - 
        ri/nu[ri])/(rout/nu[rout] - ri/nu[ri]) 
        \[Alpha]disk2[x_, lamin_, lamout_, rin_, rout_] := 
         lamin + (lamout - 
              lamin) (x/nu[x] - rin/nu[rin])/(rout/nu[rout] - rin/nu[rin])
              nu[x_] := \[Alpha] h^2 x^\[Gamma]
              vrn[x_, rout_] := -1.5 (\[Lambda]o - \[Lambda]i)/(rout/nu[rout] - 
                    ri/nu[ri]) /\[Alpha]disk[x, rout]
                    \[Xi][x_] := (x - a)/q
                    Func[x_] := 
                     T0 *((\[CapitalGamma] + 
                             1) Exp[-((\[Xi][x] - \[Beta])/\[CapitalDelta])^2] - 
                                  Exp[-((\[Xi][
                                               x] + \[Beta])/\[CapitalDelta])^2])/(\[CapitalDelta] Sqrt[
                                                     Pi])
                                                     s[x_, x0_, w_] := .5 (1 + Tanh[(x - x0)/w])
                                                     \[CapitalLambda]L[x_, f_, c_] := -f Pi (\[Mu] h^3)^2 (x/(x - a))^4
                                                     \[CapitalLambda]R[x_, f_, c_] := f Pi (\[Mu] h^3)^2 (x/(x - a))^4 
                                                     \[CapitalLambda][x_, f_, c_] := 
                                                      Piecewise[{{- f Pi (\[Mu] h^3)^2 (x/(x - a))^4, x < (a - c rh)}, {0, 
                                                          a - c rh <= x <= a + c rh}, {f Pi (\[Mu] h^3)^2 (x/(x - a))^4, 
                                                              x > (a + c rh)}}]
                                                              \[CapitalLambda]smooth[x_, f_, c_, 
                                                                w_] := (1 - s[x, a - c rh, w]) \[CapitalLambda]L[x, f, c]  + 
                                                                  s[x, a - c rh, w] s[x, a + c rh, w] \[CapitalLambda]R[x, f, c]
\[Alpha]disk[x_,rout_]:=\[Lambda]i + (\[Lambda]o-\[Lambda]i) (x/nu[x] - ri/nu[ri])/(rout/nu[rout]-ri/nu[ri]) 
\[Alpha]disk2[x_,lamin_,lamout_,rin_,rout_]:=lamin+ (lamout-lamin) (x/nu[x] - rin/nu[rin])/(rout/nu[rout]-rin/nu[rin])
nu[x_]:= \[Alpha] h^2 x^\[Gamma]
vrn[x_,rout_]:= -1.5 (\[Lambda]o-\[Lambda]i)/(rout/nu[rout]-ri/nu[ri]) /\[Alpha]disk[x,rout]
\[Xi][x_]:= (x-a)/q
Func[x_]:= T0 *((\[CapitalGamma]+1)Exp[-((\[Xi][x]-\[Beta])/\[CapitalDelta])^2] -Exp[-((\[Xi][x]+\[Beta])/\[CapitalDelta])^2])/(\[CapitalDelta] Sqrt[Pi])
s[x_,x0_,w_]:= .5(1 + Tanh[(x-x0)/w])
\[CapitalLambda]L[x_,f_,c_]:= Piecewise[{{-f a Pi (\[Mu] h^3)^2 (x/(Max[h x, Abs[x-a]]))^4,x<a},{0,x>=a}}]
\[CapitalLambda]R[x_,f_,c_]:= Piecewise[{{f a  Pi (\[Mu] h^3)^2 (a/(Max[h x, Abs[x-a]]))^4,x>a} ,{0,x<=a}}]
\[CapitalLambda][x_,f_,c_]:=Piecewise[{{- f Pi (\[Mu] h^3)^2 (x/(x-a))^4,x<(a-c rh)},{0,a-c rh <= x<= a+c rh},{f Pi (\[Mu] h^3)^2 (x/(x-a))^4,x>(a+c rh)}}]
\[CapitalLambda]smooth[x_,f_,c_,w_]:=(1-s[x,a-c h x,w])\[CapitalLambda]L[x,f,c]  + s[x,a-c h x,w] s[x,a + c h x,w] \[CapitalLambda]R[x,f,c]

sol=NDSolve[ {3 nu[r] \[Lambda]'[r] + \[Lambda][r](3 nu[r](\[Gamma]-.5)/r - PlanetFlag  (UseGaussian Func[r] + UseTorque \[CapitalLambda]smooth[r,\[CapitalLambda]f,\[CapitalLambda]c,\[CapitalLambda]w]) /(Pi  Sqrt[r])) == m[r],m'[r]== 0,\[Lambda][ri]== \[Lambda]i,\[Lambda][ro]== \[Lambda]o},{\[Lambda],m},{r,ri,ro}];


rvals = Table[ ri + 1. (i-1) (ro-ri)/(nr-1),{i,1,nr}] 
lamfinal = Table[\[Lambda][rvals[[i]]] /. sol,{i,1,nr}]

mdotfinal = m[ri] /. sol

dTr = Table[ UseGaussian Func[rvals[[i]]] + UseTorque \[CapitalLambda]smooth[rvals[[i]],\[CapitalLambda]f,\[CapitalLambda]c,\[CapitalLambda]w],{i,1,nr}]


dat = Table[{rvals[[i]],lamfinal[[i]][[1]],-mdotfinal[[1]]/(lamfinal[[i]][[1]]),dTr[[i]]},{i,1,nr}]


Export["results.dat",dat]

Print["Done"]

Exit[]


