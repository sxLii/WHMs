function dudt=prodij(Hi,Hj,Vi,Vj,xi,xj,rhoi,rhoj)
global g;
global h;

alpha=1;
beta=0.8;
garma=0.01;
flag=(Vi-Vj)*(xi-xj);
aveh=Hi/2+Hj/2;
averho=rhoi/2+rhoj/2;
avec=(sqrt(g*Hi)+sqrt(g*Hj))/2;
% phi_ij
phi=aveh*flag/((xi-xj)^2+garma*aveh^2);
% 人工粘性项
if flag<0
    vis=(-alpha*avec*phi+beta*phi^2)/averho;
else
    vis=0;
end
% 最麻烦的一项
dudt=rhoj*pi*h^2/4*gradW(xi-xj,h)*(Hj/(rhoj^2)+Hi/(rhoi^2)+vis);

end