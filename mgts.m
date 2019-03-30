function mydata = mgts(nm)
tau = 25;
for t = 0:nm-1
    if t<= tau
        s(t+1) = rand;
    else
        s(t+1) = 0.9*s(t)+(0.2*s(t-tau))/(1+s(t-tau)^10);
%         s(t)+((0.2*s(t-tau))/(1+s(t-tau)^10))-0.1*s(t);
    end
end

