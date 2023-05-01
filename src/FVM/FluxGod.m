function [flux] = FluxGod(u,v)	%Flux(U_R,U_L)
% input :[H;u] 2*1
global g;
global c;
deltaH=u(1)-v(1);
deltav=u(2)-v(2);
avev=(u(2)+v(2))/2;% 速度均值
%avec=sqrt(g*u(1))/2+sqrt(g*v(1))/2;
%cminus=(deltaH-avec/g*deltav).*abs(avev-avec)./2*[1;-g/avec];
%cplus=(deltaH+avec/g*deltav).*abs(avev+avec)./2*[1;g/avec];
cminus=(deltaH-c/g*deltav).*abs(avev-c)./2*[1;-g/c]; % C-
cplus=(deltaH+c/g*deltav).*abs(avev+c)./2*[1;g/c]; % C+

flux = (F(u,avev)+F(v,avev))/2 - (cplus+cminus)/2;
end