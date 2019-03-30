function [perf, tconv] = CheckNetwork(HiddenNodes, rStrength, nStd, PlotFlag, varargin)
% HiddenNodes = [8 6];
% rStrength = 0.9;
% nStd = 0.03;
% PlotFlag = true;
% varargin = ('trainbr');
NormalizeFunction = @(aa) aa./norm(aa);
    nbVarArgs = length(varargin);
    
    tau = 25; beta = 0.2; gamma = 0.1; step = 1; npMG = 9000;
    limit = npMG-tau-1; % determines the number of points to generate

    
    [t, x] = GenerateMackeyGlass(tau, beta, gamma, step, npMG-tau-1, false);
    
    x = x + nStd*randn(1,length(x));
    t = t + nStd*randn(1,length(t));

    npMG = length(t);

    %% Training Patterns and Targets

    tF = floor(tau/5); % Time future
    minS = npMG/10+1;
    maxS = npMG/2;
    sample = t(1,minS:maxS);
    ixMin = find(t==sample(1)); ixMax = find(t==sample(end));
    tx = [ixMin:ixMax];
%below: determining no of inputs
    nbInputs = 5;
    Xn = [];
    for i=1:nbInputs
        Xn = [Xn ; x(tx-(nbInputs-i)*tF)];
    end
    maxDelay = (nbInputs-1)*tF; % there will be a shift in the output
    T = x(tx+tF);


    r = 2/6; % proportion: Training data vs Hold-Out
    HoldOut = floor((npMG/2-npMG/10)*r);
    nData = (npMG/2-npMG/10) - HoldOut; % (npMG/2-npMG/10)*tau = maxS - minS

    % Determine data for training and testing
    testMin = nData;
    testMax = nData+HoldOut;
    shifted_x = x(minS+testMin-maxDelay:minS+testMax-maxDelay);

    % This loop allows us to get the proper network based on sigma_2
    doTrain = true;
    
    if nbVarArgs ~= 0
        trainF = varargin{1};
    else
        trainF = 'trainscg';
    end
    
    nbRepetitions = 0;
    while doTrain
        tr = []; net = [];
        net = feedforwardnet(HiddenNodes); % [nHL1 nHL2 ... nHLx]
        net.trainFcn = trainF;
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 1000;
        net.trainParam.goal = 1e-2;
        net.divideParam.trainRatio = 60/100;
        net.divideParam.valRatio = 30/100;
        net.divideParam.testRatio = 10/100;
        net.performParam.regularization = rStrength; % This helps to prevent the saturation!!!!

        [net,tr] = train(net,Xn(1,1:nData),T(1,1:nData)); %default config
        y = net(Xn(1,testMin:testMax));
        sigma_2 = var(NormalizeFunction(y)-NormalizeFunction(shifted_x));
        if sigma_2 < 0.1
            doTrain = false;
        end
        
        
        if nbRepetitions == 100
            doTrain = false;
            disp(sprintf("Time's Up: %s with std:%0.2f + nH=%d + r=%0.1f", trainF, nStd, HiddenNodes(1,length(HiddenNodes)), r));
        end
        nbRepetitions = nbRepetitions + 1;
        
    end

    perf = perform(net,y,T(1,testMin:testMax));
    tconv = tr.time(end);
        
    if PlotFlag
        
        strTP = sprintf("%dLP", length(HiddenNodes)+1);
        strTF = upper(erase(trainF,"train"));
        nHLL = length(HiddenNodes);
        nH = HiddenNodes(1,length(HiddenNodes));
        nStdP = round(nStd*100);
        rStrengthP = round(rStrength*100);
                
        lc = figure;
        plot(tr.epoch+1, tr.perf, '-o', 'DisplayName', "Training"); 
        hold on; grid on; grid minor;
        plot(tr.epoch+1, tr.tperf, '-x', 'DisplayName', "Testing");
        axis([-Inf Inf -max(max(abs(tr.tperf), abs(tr.perf))) Inf]);
        plot(tr.epoch+1, tr.goal*ones(1,length(tr.epoch)), 'g', 'DisplayName', "Goal");
        legend('show','Location','northeast');
        LocalTitle = sprintf("MG %s[%s] Performance: std=%d nHL%d=%d r=%dp", strTP, strTF, nStdP, nHLL, nH, rStrengthP);
        text(tr.epoch(end)+1, -tr.goal, sprintf('P = %0.4f',tr.perf(end)), ...
            'FontWeight', 'bold', 'HorizontalAlignment','right', 'VerticalAlignment','top');
        title(LocalTitle);
%         SaveFigure(LocalTitle);
%         close(lc);
        
        mg = figure;
        LineName = 'Target x(t+5) ';
        plot(t(1,minS+testMin:minS+testMax), NormalizeFunction(shifted_x), 'DisplayName', LineName); 
        hold on; grid on; grid minor;
        LineName = 'Output x(t+5) ';
        plot(t(1,minS+testMin:minS+testMax), NormalizeFunction(y), 'DisplayName', LineName); 
        legend('show','Location','northeast'); 
        axis([-Inf Inf -2 2]);
        LocalTitle = sprintf("MG %s[%s] Y vs T: std=%d nHL%d=%d r=%dp", strTP, strTF, nStdP, nHLL, nH, rStrengthP);
        title(LocalTitle);
%         SaveFigure(LocalTitle);
%         close(mg);
        
    end

end
