%% 
% Question : analyse weights
clear;

%Phi declaration:
phi = @(xxx) (2./(1+exp(-xxx)))-1;

%Phi prime:
phiprime = @(yyy) ((1+yyy).*(1-yyy))/2;

error = [];
alpha = 0.9;
eta = 0.05;
hidden = 3;

 x = eye(8) * 2 - 1;
% x = [1 1 1 1 1 1 1 -1];
[x_row, x_col] = size(x);
w = randn(hidden, x_row);
v = randn(x_row, hidden);
% w = randn(hidden);     %, x_row+1);
%  v = randn(x_row);     %, hidden+1);
dw = 0;
dv = 0;

steps = 0:4000; %init: 20000
for i = steps
    % Forward pass
    hin = w * x; %ones(1, x_col)];
    hout = phi(hin); %ones(1, x_col)];

    oin = v * hout;
    out = phi(oin);

    % Backward pass
    delta_o = (out - x) .* phiprime(out);
    delta_h = (v' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :);

    % Weight update
%     dw = (dw .* alpha) - (delta_h * [x; ones(1, x_col)]') .* (1 - alpha);
    dw = (dw .* alpha) - (delta_h * x') .* (1 - alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1 - alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    error = [error; sum(sum(abs(sign(out) - x) ./ 2))]; %sign= change (-) vals to be -1, positive to be 1
end

sign(out);
plot(steps, error);