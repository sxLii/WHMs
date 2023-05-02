% 假设阀门瞬间关闭
clear
tic
L=300;                          % 管线长度
Hr=70;                          % 泵压力
N=10;                           % 分段数
NS=N+1;                         % 节点数
e=0.001651;                     % 壁厚m,0.065''
D=0.00635-2*e;	                % 管道内径
K=2.1e+9;                       % 流体体积弹性系数
Rho=1000;	                    % 液体密度kg/m^3
E=2.1e11;                       % 弹性模数
G=9.806;                        % 重力加速度
f=0.018;                        % 摩擦系数
A=pi*(D^2)/4;                     % 管道横截面积

% 网格划分
dx=L/N;                         % 单位长度
t_max=20;                       % 总计算时间
a=sqrt(K/Rho/(1+K*D/(E*e)));     % 波速
dt=dx/a;                        % 单位计算时间

k=1;
t=0;time(k)=t;

% 常数B和R
B=a/(G*A);
R=f*dx/(2*G*D*(A^2));

% 第一列流量和压力
u0=0.1; % initial volecity

Q(k,1:NS)=0; % A*u0;
H(k,2:NS)=0;
H(1,1)=Hr;

while t<t_max
   for j=2:N
        CP=H(k,j-1)+B*Q(k,j-1)-R*Q(k,j-1)*abs(Q(k,j-1));    % +dt*Q(k,j-1)/A;    % CP==C-==CR
        CM=H(k,j+1)-B*Q(k,j+1)+R*Q(k,j+1)*abs(Q(k,j+1));    % -dt*Q(k,j+1)/A;    % CM==C+==CL
        H(k+1,j)=(CP+CM)/2;                                 % 计算压头
        Q(k+1,j)=(H(k+1,j)-CM)/(B);                         % 计算流量
         
   end
    
    % 边界值
    CP=H(k,N)+B*Q(k,N)-R*Q(k,N)*abs(Q(k,N));    % +dt*Q(k,N)/A;    % 单独计算CP
    CM=H(k,2)-B*Q(k,2)+R*Q(k,2)*abs(Q(k,2));    % -dt*Q(k,2)/A;    % 单独计算CM

    H(k+1,1)=Hr;                                % 压头第一列
    Q(k+1,1)=(H(k+1,1)-CM)/B;                   % 流量第一列数值，边界条件公式
    
    Q(k+1,NS)=0;                                % 流量最后一列
    H(k+1,NS)=CP;                               % 压头最后一列数值，边界条件公式

    t=t+dt;
    k=k+1;
    time(k)=t;
    
end
toc

plot(time,H(:,N+1))
title('MOC-阀门处圧力曲线');
xlabel('单位：s');
ylabel('单位：m');