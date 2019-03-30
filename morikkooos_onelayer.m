clc, close all;

classA(1,:) = randn(1, 100) .* 0.5 + 0.5; %init0.5
classA(2,:) = randn(1, 100) .* 0.5 + 0.5;
classB(1,:) = randn(1, 100) .* 0.5 - 1.0;%init 1
classB(2,:) = randn(1, 100) .* 0.5 + 0.0;

patterns = [classA, classB];
targets = [ones(1, 100), -ones(1, 100)];

permute = randperm(200);
% Permute data
patterns = patterns(:, permute);
targets = targets(:, permute);

[insize, ndata] = size(patterns);
[outsize, ndata] = size(targets);

 if targets>0
 end
 
plot (patterns(1, find(targets>0)), ...
patterns(2, find(targets>0)), '*', ...
patterns(1, find(targets<0)), ...
patterns(2, find(targets<0)), '+');

eta = 0.001;
w = randn(1, 3);
X = [patterns; ones(1, ndata)];
epoch = 0;
for i = 0:60
    % Compute 3 times
    for j = 0:3
        deltaW = -eta*( w*X - targets)*X';
        w = w + deltaW;
    end
    
    % Show
    p = w(1, 1:2);
    k = -w(1, insize+1) / (p*p');
    l = sqrt(p*p');
    axis([-2, 2, -2, 2], 'square'); %determine size of the images
    
    plot (patterns(1, find(targets>0)), ...          
        patterns(2, find(targets>0)), '*', ...
        patterns(1, find(targets<0)), ...
        patterns(2, find(targets<0)), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-');
    txt = ['Epoch ', num2str(epoch)];
    text(-2.5,1.7,txt ,'Color','red','FontSize',14);
    drawnow;
    epoch = epoch + 1;
     %error = [error; sum(sum(abs(sign(out) - targets) ./ 2))];
    pause(0.1);

    % Wait user
    %waitforbuttonpress;
end