clear;

out = [];
%Phi declaration:
phi = @(xxx) (2./(1+exp(-xxx)))-1;

%Phi prime:
phiprime = @(yyy) ((1+yyy).*(1-yyy))/2;

% Create original function
x_full=[-5:1:5]';
y_full=x_full;
z=exp(-x_full.*x_full*0.1) * exp(-y_full.*y_full*0.1)' - 0.5;
mesh( x_full, y_full, z);
% moriko's
[z_row, z_col] = size(z);
ndata = z_row*z_col;
%


% Parse ann value
targets = reshape(z, 1, ndata);
[xx, yy] = meshgrid(x_full, y_full);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

% Randomize and select the data
permute = randperm(ndata);
% Permute data
x = patterns(:, permute);
y = targets(:, permute);
% Number of points that are kept
n = 25;
x = x(:, 1:n);
y = y(:, 1:n);
[p_row p_col] = size(x);

% Initialize ANN value
error = [];
alpha = 0.9;
eta = 0.15;
hidden = 10;
w = randn(hidden, p_row+1);
v = randn(1, hidden+1);
dw = 0;
dv = 0;

steps = 0:500;
for i = steps
    % Forward pass training
    hin = w * [x; ones(1, p_col)];
    hout = [phi(hin); ones(1, p_col)];

    oin = v * hout;
    out = phi(oin);

    % Backward pass
    delta_o = (out - y) .* phiprime(out);
    delta_h = (v' * delta_o) .* phiprime(hout);
    delta_h = delta_h(1:hidden, :);

    % Weight update
    dw = (dw .* alpha) - (delta_h * [x; ones(1, p_col)]') .* (1 - alpha);
    dv = (dv .* alpha) - (delta_o * hout') .* (1 - alpha);
    w = w + dw .* eta;
    v = v + dv .* eta;
    
    error = [error ; sum((y - out).^2) / sum(y .^2)];
    
    

end
figure(2);
plot(steps, error);

figure(1);
    % Forward pass test
    hin = w * [patterns; ones(1, ndata)];
    hout = [phi(hin); ones(1, ndata)];
    
    oin = v * hout;
    out = phi(oin);
    zz = reshape(out, z_row, z_col);
    mesh(x_full, y_full, zz);
    axis([-5 5 -5 5 -0.7 0.7]);
    drawnow;