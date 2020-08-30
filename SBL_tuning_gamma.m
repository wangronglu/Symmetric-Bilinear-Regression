function [alpha_final,Lambda_final,Beta_final,coefM_final,coefM_final_vec,min_MSE_test,gamma_seq,MSE_set] = ...
    SBL_tuning_gamma(W_train,y_train,W_test,y_test,K,nreps,ngam,gamma_min_ratio,maxit,fullit,tol)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SBL_tuning_gamma.m automatically find gamma_max that penalizes all the
% coefficients to zero, and select the optimal gamma value producing the
% smallest MSE on test set. It returns the estimated coefficients under the
% optimal gamma.

% Input:
%   W_train: VxVxn_train, adjacency matrices of subjects in training set.
%   y_train: n_train x 1 vector, response of subjects in training set.
%   W_test: VxVxn_test, adjacency matrices of subjects in test set.
%   y_test: n_test x 1 vector, response of subjects in test set.
%   K: number of components in SBL
%   nreps: number of random initializations
%   ngam: number of gamma values
%   gamma_min_ratio: gamma_min = gamma_min_ratio * gamma_max
%   maxit: maximum iterations (>=2),e.g. maxit=1000.
%   fullit: number of iterations that cycle all variables; after that the 
%           algorithm only updates active variables.
%           ( 2 <= fullit <= maxit, e.g. fullit = 100)
%   tol: tolerance of relative change in objective function,e.g. tol=1e-5.
%
% Output:
%   alpha_final: scalar, estimate of alpha under the optimal gamma
%   Lambda_final: Kx1 vector, estimate of Lambda under the optimal gamma
%   Beta_final: VxK matrix, estimate of Beta under the optimal gamma
%   coefM_final: VxV matrix, Beta_final * diag(Lambda_final) * Beta_final'
%   coefM_final_vec: 2 x upper triangular of coefM_final 
%   min_MSE_test: minimum MSE of test set
%   gamma_seq: gamma sequence for tuning
%   MSE_set: set of MSEs on test set across gamma values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

V = size(W_train,1);
n_test = length(y_test);

%% set sequence of gamma values
% choose upper bound
gamma = 1;

for rep = 1:nreps
    % initialization
    rng(rep-1, 'twister')
    Beta_ini = 1 - 2 * rand(V,K); % U(-1,1)
    
    [~,~,Beta] = SBL1(y_train,W_train,V,K,gamma,maxit,fullit,tol,Beta_ini);
    
    if nnz(Beta)>0
        break
    end
end

while( nnz(Beta)>0 )
    gamma = 2 * gamma;
    
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 1 - 2 * rand(V,K); % U(-1,1)
        
        [~,~,Beta] = SBL1(y_train,W_train,V,K,gamma,maxit,fullit,tol,Beta_ini);
        
        if nnz(Beta)>0
            break
        end
    end
end

while( nnz(Beta)==0 )
    gamma = 0.5 * gamma;
    
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 1 - 2 * rand(V,K); % U(-1,1)
        
        [~,~,Beta] = SBL1(y_train,W_train,V,K,gamma,maxit,fullit,tol,Beta_ini);
        
        if nnz(Beta)>0
            break
        end
    end    
end

gamma_max = 2 * gamma;

% set lower bound 
gamma_min = gamma_min_ratio * gamma_max;

% set gamma sequence 
gamma_seq = exp(linspace(log(gamma_min), log(gamma_max), ngam));

%% tuning on test data
MSE_set = zeros(ngam,1);

for j=1:ngam
    disp(['j=',num2str(j)])
    
    gamma = gamma_seq(j);
    
    LFmin = Inf;
    for rep = 1:nreps
        % initialization
        rng(rep-1, 'twister')
        Beta_ini = 1 - 2 * rand(V,K); % U(-1,1)
        
        [alpha_cand,Lambda_cand,Beta_cand,LF_cand] = ...
            SBL1(y_train,W_train,V,K,gamma,maxit,fullit,tol,Beta_ini);
        
        if (LF_cand(end) < LFmin)
            LFmin = LF_cand(end);
            alpha = alpha_cand;
            Beta = Beta_cand;
            Lambda = Lambda_cand;
        end
    end
    
    % compute MSE
    coefM = (Beta.* repmat(Lambda',[V,1]))* Beta'; % V x V
    y_pred = alpha + squeeze(sum(sum(repmat(coefM,[1,1,n_test]).* W_test,1),2));
    
    MSE_set(j) = sum((y_test - y_pred).^2)/n_test;
end

%% select optimal tuning parameter
[min_MSE_test,ind_opt] = min(MSE_set); 
gamma_opt = gamma_seq(ind_opt);

% estimate model at optimal penalty factor with training data
LFmin = Inf;
for rep = 1:nreps
    % initialization
    rng(rep-1, 'twister')
    Beta_ini = 1 - 2 * rand(V,K); % U(-1,1)
    
    [alpha_cand,Lambda_cand,Beta_cand,LF_cand] = ...
        SBL1(y_train,W_train,V,K,gamma_opt,maxit,fullit,tol,Beta_ini);
    
    if (LF_cand(end) < LFmin)
        LFmin = LF_cand(end);
        alpha = alpha_cand;
        Beta = Beta_cand;
        Lambda = Lambda_cand;
        % disp(['rep=',num2str(rep)])
    end
end

% estimate model at selected penalty factors with full data ----------
W = cat(3,W_train,W_test);
y = [y_train; y_test];

clear W_train W_test

% use estimates from training data under optimal penalty factor as initial
% value
[alpha_final,Lambda_final,Beta_final]...
    = SBL1(y,W,V,K,gamma_opt,maxit,fullit,tol,Beta);

coefM_final = (Beta_final.* repmat(Lambda_final',[V,1]))* Beta_final';

% extract upper-triangular part
UTidx = triu(true(V),1); % excluding diagonal
coefM_final_vec = 2 * coefM_final(UTidx);
