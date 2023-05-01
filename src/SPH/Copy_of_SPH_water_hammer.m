clear
tic

global a;
global h;
global g;

L=300;                          % 管线长度
Hr=70;                          % 泵压力
N=10;                           % 分段数,即粒子数
NS=N+4;                         % 节点数 N+1+2两边各两个粒子
e=0.001651;                     % 壁厚m,0.065''
D=0.00635-2*e;	                % 管道内径
K=2.1e+9;                       % 流体体积弹性系数
Rho=1000;	                    % 液体密度kg/m^3
E=2.1e11;                       % 弹性模数
g=9.806;                        % 重力加速度
f=0.018;                        % 摩擦系数
A=pi*(D^2)/4;                     % 管道横截面积

% 网格划分
CFL=1;
dx=L/N;                         % 单位长度
t_max=20;                       % 总计算时间
a=sqrt(K/Rho/(1+K*D/(E*e)));     % 波速
dt=CFL*dx/a;                        % 单位计算时间
k=1;
t=0;time(k)=t;

% particle initial
h=1.3*dx; % kernal of sph
A_particle=pi*h^2/4;
m=Rho/N;
rho0(k,1:NS)=m/A_particle;
rho(k,1:NS)=0;
x(k,1:NS)=0-dx*3/2:dx:L+3/2*dx;

% 3~N+2
V(k,1:NS)=0;
H(k,3:NS)=0;
H(k,1:2)=Hr;
% ans
V_ans(k,1)=0;
V_ans(k,2:N+1)=V(k,3:N+2);
V_ans(k,N+2)=0;
H_ans(k,1)=Hr;
H_ans(k,2:N+1)=H(k,3:N+2);
H_ans(k,N+2)=0;


while t<t_max
    % update density
    for j=3:N+2
        rho(k,j)=A_particle*(rho0(k,j-2)*W(x(k,j)-x(k,j-2),h)...
            +rho0(k,j-1)*W(x(k,j)-x(k,j-1),h)...
            +rho0(k,j+1)*W(x(k,j)-x(k,j+1),h)...
            +rho0(k,j+2)*W(x(k,j)-x(k,j+2),h)...
            +rho0(k,j)*W(0,h));
    end
    rho(k,1:2)=rho(k,3); % dummy
    rho(k,N+3)=rho(k,N+2); % mirror
    rho(k,N+4)=rho(k,N+1);
    rho0(k+1,:)=rho(k,:);

    % update time
    for j=3:N+2

        H(k+1,j)= H(k,j)-dt*a^2/g*(...
            rho(k,j-2)*A_particle*(V(k,j-2)-V(k,j))*gradW(x(k,j)-x(k,j-2),h)...
            +rho(k,j-1)*A_particle*(V(k,j-1)-V(k,j))*gradW(x(k,j)-x(k,j-1),h)...
            +rho(k,j+1)*A_particle*(V(k,j+1)-V(k,j))*gradW(x(k,j)-x(k,j+1),h)...
            +rho(k,j+2)*A_particle*(V(k,j+2)-V(k,j))*gradW(x(k,j)-x(k,j+2),h)...
            );

        V(k+1,j)=V(k,j)+dt*(-rho(k,j)*g*(...
            prodij(H(k,j),H(k,j-2),V(k,j),V(k,j-2),x(k,j),x(k,j-2),rho(k,j),rho(k,j-2))...
            +prodij(H(k,j),H(k,j-1),V(k,j),V(k,j-1),x(k,j),x(k,j-1),rho(k,j),rho(k,j-1))...
            +prodij(H(k,j),H(k,j+1),V(k,j),V(k,j+1),x(k,j),x(k,j+1),rho(k,j),rho(k,j+1))...
            +prodij(H(k,j),H(k,j+2),V(k,j),V(k,j+2),x(k,j),x(k,j+2),rho(k,j),rho(k,j+2))...
            ) +g*f*V(k,j)*abs(V(k,j))/2/D...
            );

        x(k+1,j)=x(k,j)+dt*(V(k+1,j)+V(k,j))/2;

    end

    % update buffer
    H(k+1,1:2)=Hr; V(k+1,1:2)=V(k+1,3); % dummy
    x(k+1,1:2)=x(k,1:2)+dt*(V(k+1,1:2)+V(k,1:2))/2;

    H(k+1,N+3)=-H(k+1,N+2); V(k+1,N+3)=-V(k+1,N+2);% mirror1
    H(k+1,N+4)=-H(k+1,N+1); V(k+1,N+4)=-V(k+1,N+1);% mirror2
    x(k+1,N+3:N+4)=x(k,N+3:N+4)+dt*(V(k+1,N+3:N+4)+V(k,N+3:N+4))/2;
    
    % 边界值
    V_ans(k+1,2:N+1)=V(k+1,3:N+2);
    H_ans(k+1,2:N+1)=H(k+1,3:N+2);
    
    H_ans(k+1,1)=Hr;
    V_ans(k+1,1)=V_ans(k+1,2)+(V_ans(k,1)-V_ans(k,2))*g/a;
    
    V_ans(k+1,N+2)=0;
    H_ans(k+1,N+2)=H_ans(k+1,N+1)+a/g*V_ans(k+1,N+1);

    t=t+dt;
    k=k+1;
    time(k)=t;

end
toc

plot(time,H_ans(:,N+2))
title('MOC-阀门处圧力曲线');
xlabel('单位：s');
ylabel('单位：m');