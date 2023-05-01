function [Source] = S(U)
% input :[H;u] 2*1
    global g;
    global f;
    global D;
    Source(1,1)=0;
    Source(2,1)=-g*f*(U(2)^2)/(2*D);
end