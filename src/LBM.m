clear; close all; clc;
tic
% initial
L=500;
f=0.02;
a=1000;
Hr=400;
V0=3.25996;
D=2;
g=9.81;

% mesh
nodes=200;
timesteps = 400;
dx=1;
dt=1; 

% lambda
lambdax=L/nodes;
lambdat=lambdax/a;
lambdau=a;
lambdah=a/g*lambdau;
lambdaf=1/(lambdat*lambdau);
ff=f/lambdaf;
HrHr=Hr/lambdah;
DD=D/lambdax;


% Derived inputs.
omega = dt/1;

% Initial conditions.
f0 = zeros(1,nodes);
f1 = zeros(1,nodes);
f2 = zeros(1,nodes);

k=1;
vv(k,1:nodes)=V0/lambdau;
hh(k,2:nodes)=HrHr;
hh(k,1)=HrHr;

% tmie
t=0;
time(k)=t;
% Main loop.
for iter = 1:timesteps
    
    % Collision.
    feq0 = 0;
    feq1 = vv(k,:)+hh(k,:)./2;
    feq2 = vv(k,:)-hh(k,:)./2; 

    R(1,1:nodes)=ff.*vv(k,:).*abs(vv(k,:))/(4*DD);

    f0=(1-omega)*f0+omega*feq0;
    f1 = (1-omega)*f1 + omega*feq1-R;
    f2 = (1-omega)*f2 + omega*feq2-R;
    % Streaming.
    f1(2:end) = f1(1:end-1);
    f2(1:end-1) = f2(2:end);
    % BC.
    f1(1)=HrHr+f2(1);
    f2(end)=vv(k,end)-f1(end)-f0(end);

    % Reconstruct solution.
    vv(k+1,:)= f0 + f1 + f2;
    hh(k+1,:)= f1 - f2;
    k=k+1;

    % time
    t=t+dt;
    time(k)=t;
end
vv(k+1,:) = f0 + f1 + f2;
hh(k+1,:) = f1 - f2;
t=t+dt;
time(k+1)=t;
time=time';
h=hh*lambdah;
v=vv*lambdau;
toc
% Plot results.
plot(time(:),hh(:,end-1))
title('Solution');
xlabel('t');
ylabel('h');