% func B-Spline_w 是B-样条函数(光滑核函数). 
function w = W(r,h)
    r = abs(r);
    alpha = 1/h;
    q = r/h;
    if q>=0 && q<1
        w = alpha*(2/3-q^2+0.5*q^3);
    elseif q>=1 && q<2
        w = alpha*(1/6*(2-q)^3);
    elseif q>=2
        w = 0;
    else
        fprintf("q = %d 是否小于了0; 请检查! ",q);
    end
end