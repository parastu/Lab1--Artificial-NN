%Testing the single layer only!!

clc, clear all, close all

n_point=5;
%sse = @(xxx) xxx(1,1)^2 + xxx(1,2)^2 + xxx(1,3)^2;

xa1 = randn(1,n_point);
xa2 = randn(1,n_point);
XA (:,:) = [xa1;xa2];
ta = ones(1, n_point);

xb1 = randn(1,n_point);
xb2 = randn(1,n_point);
XB (:,:) = [xb1;xb2];
tb = zeros(1, n_point);
 %perform shuffle
 %belum di shuffle
XAB = [XA,XB]; %ini inputnya
Tab = [ta,tb]; %ini targetnya
na=size(XA,2);
nb=size(XB,2);
n_total = na + nb;
%adding bias to the input and weight
X=[XAB;ones(1,n_total)];    %the input which is usefull
%determining the weight:
W_old = randn(1,3); %note: change the '3' part...
% mu = 0.001;
% ii = 100;
err_val = 100;
epoch = 0;
%the single layer
while err_val > 0.001 && epoch > 10000
    err_val = 0.0;
    epoch = epoch + 1
    for i = 1:n_total
        x_tempp = X(:,i);
        t_tempp = Tab(:,i);
        temp_n = W_old*x_tempp;
        tresh_n = hardlim(temp_n);
        err_val(:,i) = tresh_n - t_tempp;
        X_trans = X';
        %sseval = sse(err_val)
%         if err_val >(0.001/n_total)
            %ii = sum(err_val)
            delta_W = (err_val(:,i))*X';  %%%%%% MU BELUM MASUKKKKK!!!!
            W_new = W_old + delta_W;
            W_old = W_new;
%         end
        %err_val = err_val +1
        
    end
end




% 
% % Define class 1
% mean_a1 = -8; 
% mean_a2 =  8;
% std_a1  = 2;
% std_a2 = 2;
% 
% % Implement class 1
% xa1 = mean_a1+std_a1*randn(1,n_point);
% xa2 = mean_a2+std_a2*randn(1,n_point);
% XA(:,:) =[xa1;xa2];
% ta = ones(1,n_point); % label of class one = 1
% 
% % Define class 2
% mean_b1 = 2; 
% mean_b2 =  2;
% std_b1  = 2;
% std_b2 = 2;
% 
% % Impleent class 2
% xb1=mean_b1+std_b1*randn(1,n_point);
% xb2=mean_b2+std_b2*randn(1,n_point);
% 
% XB(:,:) =[xb1;xb2];
% tb = zeros(1,n_point);
% %%
% na=size(XA,2);
% nb=size(XB,2);
% 
% ntot=na+nb;
% 
% t_tot=[ta,tb];
% 
% shuf=randperm(ntot);
% XAB=[XA,XB];
% 
% X_temp= XAB(:,shuf);
% 
% t_tot=t_tot(shuf);