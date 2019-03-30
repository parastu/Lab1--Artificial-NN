function [perf, tconv] = CheckNetwork_newest(HiddenNodes, rStrength, nStd, PlotFlag, varargin)
%    NormalizeFunction = @(aa) aa./norm(aa);
 close all;   
NormalizeFunction = @(bb) (bb-mean(bb))/(max(bb-mean(bb)));
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

        [net,tr] = train(net,Xn(:,1:nData),T(1,1:nData)); %default config
        y = net(Xn(:,testMin:testMax));
        sigma_2 = var(NormalizeFunction(y)-NormalizeFunction(x(minS+testMin:minS+testMax)))
        plot(t(1,minS+testMin:minS+testMax), y)
        if sigma_2 < 0.1
            doTrain = false;
        end
        
        
        if nbRepetitions == 100
            doTrain = false;
            disp(sprintf("Stopped! %s with std:%0.2f + nH=%d + r=%0.1f", trainF, nStd, HiddenNodes(1,length(HiddenNodes)), r));
        end
        nbRepetitions = nbRepetitions + 1;        
    end
    
    nbRepetitions

    perf = perform(net,y,T(1,testMin:testMax));
    tconv = tr.time(end);
        
    if PlotFlag
        
        strTP = sprintf("%dLayers", length(HiddenNodes)+1);
        %% det name of function/bayesian etc
    if upper(erase(trainF,"train")) == "BR"
            strTF = 'Bayesian Regulation';
    elseif upper(erase(trainF,"train")) == "LM"
            strTF = 'Levenberg-Marquardt';
    elseif upper(erase(trainF,"train")) == "BFG"
            strTF = 'BFGS Quasi-Newton';
        elseif upper(erase(trainF,"train")) == "SCG"
            strTF = 'Scaled Conj. Grad';
             elseif upper(erase(trainF,"train")) == "CGB"
            strTF = 'Conj. Grad with Powell/Beagle Restarts';
             elseif upper(erase(trainF,"train")) == "CGF"
            strTF = 'Fletcher-Powell Conj. Grad';
             elseif upper(erase(trainF,"train")) == "CGP"
            strTF = 'Polak-Ribiere Conj. Grad';
             elseif upper(erase(trainF,"train")) == "OSS"
            strTF = 'One Step Secant';
             elseif upper(erase(trainF,"train")) == "DX"
            strTF = 'Var Learning Rate Grad. Desc';
             elseif upper(erase(trainF,"train")) == "DM"
            strTF = 'Grad. Desc. w/ Momentum';
             elseif upper(erase(trainF,"train")) == "GD"
            strTF = 'Grad. Descent';
        else
            strTF = 'other';
    end
        %%
% orig code:        strTF = upper(erase(trainF,"train"));
        nHLL = length(HiddenNodes);
        nH = HiddenNodes(1,length(HiddenNodes));
        nStdP = round(nStd*100);
        rStrengthP = round(rStrength*100);
                
        lc = figure;
        plot(tr.epoch+1, tr.perf,'-o','Color','b', 'DisplayName', "Training"); 
        hold on; grid on; grid minor;
        plot(tr.epoch+1, tr.tperf, '-x','Color','m', 'DisplayName', "Testing");
        axis([-Inf Inf -max(max(abs(tr.tperf), abs(tr.perf))) Inf]);
        plot(tr.epoch+1, tr.goal*ones(1,length(tr.epoch)), 'Color','k', 'DisplayName', "Goal");
        legend('show','Location','southwest');
%         title(sprintf("Mack-Glass %s[%s] Param: std=%d nHL%d=%d r=%dp", strTP, strTF, nStdP, nHLL, nH, rStrengthP));
        title(sprintf("Mack-Glass %s[%s] ", strTP, strTF));

%         text(tr.epoch(end)+1, -tr.goal, ...
%             'FontWeight', 'bold', 'HorizontalAlignment','right', 'VerticalAlignment','top');
         text(tr.epoch(end)+1, -tr.goal, sprintf('P = %0.4f',tr.perf(end)), ...
            'FontWeight', 'bold', 'HorizontalAlignment','right', 'VerticalAlignment','top');

%         text(sprintf("Param: std=%d nHL%d=%d r=%dp",nStdP, nHLL, nH, rStrengthP));
%         title(LocalTitle);

        
        mg = figure;
        LineName = 'Target';
        plot(t(1,minS+testMin:minS+testMax), NormalizeFunction(x(1,minS+testMin:minS+testMax)),'Color','m', 'DisplayName', LineName); 
        hold on; grid on; grid minor;
        LineName = 'Output';
        plot(t(1,minS+testMin:minS+testMax), NormalizeFunction(y),'Color','k', 'DisplayName', LineName); 
        legend('show','Location','southwest'); 
        axis([-Inf Inf -2 2]);
%         LocalTitle = sprintf(" %s[%s] Out - Target:std=%d nHL%d=%d r=%dp", strTP, strTF, nStdP, nHLL, nH, rStrengthP);
LocalTitle = sprintf(" %s[%s] Out - Target", strTP, strTF);
%         text(0.5,0.02, "std=%d nHL%d=%d r=%dp",nStdP, nHLL, nH, rStrengthP);
%         text(2, 0, 'outside', 'clipping', 'off');
        title(LocalTitle);
%         SaveFigure(LocalTitle);
%         close(mg);
        
    end


end
