% 假设阀门瞬间关闭
clear
tic
global g;
global f;
global D;
global c;

L=300;                          % 管线长度
Hr=70;                          % 泵压力
N=25;                           % 分段数
NS=N+1;                         % 节点数
e=0.001651;                     % 壁厚m,0.065''
D=0.00635-2*e;	                % 管道内径
K=2.1e+9;                       % 流体体积弹性系数
Rho=1000;	                    % 液体密度kg/m^3
E=2.1e11;                       % 弹性模数
g=9.806;                        % 重力加速度
f=0.018;                        % 摩擦系数
A=pi*(D^2)/4;                     % 管道横截面积

% 网格划分
CFL=.5;
dx=L/N;                         % 单位长度
t_max=20;                       % 总计算时间
c=sqrt(K/Rho/(1+K*D/(E*e)));     % 波速
dt=CFL*dx/c;                        % 单位计算时间
lambda=dt/dx;
u0=0.1;
k=1;
t=0;time(k)=t;

% 第一行压力和第二行为流速
U(k,2:NS)=Hr-(2:NS)*f*u0*u0*dx/(L*2*D);
%U(k,2:NS)=0;
U(k,1)=Hr;
U(k+1,1:NS)=u0;

while t<t_max
    for j=2:N
        U(k+2:k+3,j)=U(k:k+1,j)+lambda*S(U(k:k+1,j))...
            -lambda*(FluxGod(U(k:k+1,j+1),U(k:k+1,j))...
            -FluxGod(U(k:k+1,j),U(k:k+1,j-1)));
    end

    % 边界值

    U(k+2,1)=Hr;                                % 压头第一列
    U(k+3,1)=U(k+3,2)+g/c*(U(k+2,1)-U(k+2,2));   % 流量第一列数值，边界条件公式

    U(k+3,NS)=0;                                % 流量最后一列
    U(k+2,NS)=U(k+2,NS-1)+c/g*(U(k+3,NS-1)-U(k+3,NS));                               % 压头最后一列数值，边界条件公式

    lambda=dt/dx;
    t=t+dt;
    k=k+2; % 一行为压力，一行为流速
    time((k-1)/2)=t;

end
toc

plot(time,U(1:2:end-2,N+1))
title('FVM-阀门处圧力曲线');
xlabel('单位：s');
ylabel('单位：m');