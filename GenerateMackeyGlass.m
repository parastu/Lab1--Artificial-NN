function [t, x] = GenerateMackeyGlass(tau, beta, gamma, step, limit, PlotFlag)

    t = [-tau:step:limit*step];
    x = zeros(1,length(t));
    x(find(t==0))=1.5;
    for i=find(t==0):find(t==t(end))-1
        x(1,i+1) = x(1,i)+(beta*x(1,i-tau))/(1+(x(1,i-tau))^10)-gamma*x(1,i);
    end
%    NormalizeFunction = @(aa) aa./norm(aa);
   NormalizeFunction = @(bb) (bb-mean(bb))/(max(bb-mean(bb)));
   
x = NormalizeFunction(x);
%x = x.*fspecial('gaussian', size(x),1);
% x = imnoise(x,'gaussian',0,0.1);
%     x = x./norm(x)
    
    if PlotFlag
        plot(t, x)
        axis([0 Inf -Inf Inf]);
    end
    
end



