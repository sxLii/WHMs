function [Flux] = F(U,avev)
  % input :[H;u] 2*1
  global g;
  global c;
  
  Flux(1,1)=avev*U(1)+c*c/g*U(2);
  Flux(2,1)=g*U(1)+avev*U(2);
end