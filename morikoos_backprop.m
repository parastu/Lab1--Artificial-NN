clc, clear all, close all;

%n sep dataa
classA(1,:) = [randn(1,50) .* 0.2 - 1.0 ...
                randn(1,50) .* 0.2 + 1.0];
classA(2,:) = randn(1,100) .* 0.2 + 0.3;
classB(1,:) = randn(1,100) .* 0.3 + 0.0;
classB(2,:) = randn(1,100) .* 0.3 - 0.1;


%setdaataaa to plot the graph

% Put all the points together and plot them
patterns = [classA, classB];
targets = [ones(1, 100), -ones(1, 100)];

permute = randperm(200);
% Permute data
patterns = patterns(:, permute);
targets = targets(:, permute);

[insize, ndata] = size(patterns);
[outsize, ndata] = size(targets);

plot (patterns(1, find(targets>0)), ...
patterns(2, find(targets>0)), '*', ...
patterns(1, find(targets<0)), ...
patterns(2, find(targets<0)), '+');

%backprop

X = [patterns; ones(1, ndata)];
alpha = 0.9;

hidden = 5;
w = randn(hidden, insize+1);
v = randn(1, hidden+1);
dw = 0;
dv = 0;

eta = 0.01;
error = [];
steps = 0:500;

%Phi declaration:
phi = @(xxx) (2./(1+exp(-xxx)))-1;

%Phi prime:
phiprime = @(yyy) ((1+yyy).*(1-yyy))/2;

epoch = 0;
for i = steps
    % Forward pass
    hin = w * [patterns; ones(1, ndata)]; %ini W*X
    hout = [phi(hin); ones(1, ndata)];

    oin = v * hout;
    out = phi(oin);

    % Backward pass
    delta_o = (out - targets) .* phiprime(out);
    delta_h = (v' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :);

    % Weight update
    dw = (dw .* alpha) - (delta_h * X') .* (1 - alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1 - alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    %drawing
    % Show
      %only w
    p = w(1, 1:2);
    k = -w(1, insize+1) / (p*p');
    l = sqrt(p*p');
      %only v
     pp = v(1, 1:2);
    kk = -v(1, insize+1) / (pp*pp');
    ll = sqrt(pp*pp'); 
    
    axis([-2, 2, -2, 2], 'square'); %determine size of the images
%     drawnow;
    plot (patterns(1, find(targets>0)), ...          
        patterns(2, find(targets>0)), '*', ...
        patterns(1, find(targets<0)), ...
        patterns(2, find(targets<0)), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-',...
       [pp(1), pp(1)]*kk + [-pp(2), pp(2)]/ll, ...
        [pp(2), pp(2)]*kk + [pp(1), -pp(1)]/ll, '--' );
     txt = ['Epoch ', num2str(epoch)];
    text(-1.5,-0.2,txt ,'Color','red','FontSize',14);
    epoch = epoch + 1;
     drawnow;
     
    error = [error; sum(sum(abs(sign(out) - targets) ./ 2))];
end

figure(2);
plot(steps, error);