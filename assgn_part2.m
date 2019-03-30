% mydata = mgts(nm);
tau = 25;
nm = 3000;
for t = 0:nm-1
    if t<= tau
        s(t+1) = rand;
    else
        s(t+1) = s(t)+(0.2*s(t-tau))/(1+s(t-tau)^10) - 0.1*s(t);
    end
    %input = [s(t-20); s(t-15);s(t-10);s(t-5);s(t)];
    %target = s(t+5)
end


mgs = s;

mydata = mgs(301:1500);
t_use=301:1500;
use_data = [t_use; mydata+1];
% use_data = [mydata];
figure(2)
%[aa, bb] = mydata;
plot(t_use, mydata);


input = [s(t-20); s(t-15);s(t-10);s(t-5);s(t)];
% target = s(t+5);
% print s(t+5)



