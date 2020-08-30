function [alpha,Lambda,Beta,LF] = SBL1(y,W,V,K,gamma,maxit,fullit,tol,Beta_ini)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SBL1 fits symmetric bilinear regression with L1 constraint
% y_i = \alpha + \sum_{h=1}^K lambda_h * t(\beta_h) * W_i * \beta_h + \epsilon_i
%
% Input:
%   y: n-by-1 response vector
%   W: VxVxn array, W(:,:,i) is symmetric with zero diagonal 
%   V: the number of nodes in the network 
%   K: number of components in SBL
%   gamma: penalty factor of L1 norm
%   maxit: maximum iterations (>=2),e.g. maxit=1000.
%   fullit: number of iterations that cycle all variables; after that the 
%           algorithm only updates active variables.
%           ( 2 <= fullit <= maxit, e.g. fullit = 100)
%   tol: tolerance of relative change in objective function,e.g. tol=1e-5.
%   Beta_ini: VxK matrix, initial value of Beta
%
% Output:
%   alpha: (scalar) intercept of regression
%   Lambda: Kx1 vector
%   Beta: VxK matrix
%   LF: values of loss function across iterations
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(y);
Beta = Beta_ini;
LF = zeros(maxit,1);
        
% initialize alpha and Lambda, compute initial LF value ------------------
% detect degenerate components
act_comp = sum(Beta~=0,1) > 1; % 1 x K
nact = sum(act_comp);
Beta(:,~act_comp) = zeros(V,K - nact);

bWb = zeros(n,K); % beta_h'* W_i * beta_h
bbt_sum = zeros(K,1); % sum of |LT(beta_h * beta_h')|
LTidx = tril(true(V),-1); % excluding diagonal

for h=1:K
    if act_comp(h)
        bbt_h = Beta(:,h) * Beta(:,h)'; % V x V
        bWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* W,1),2)); 
        bbt_sum(h) = sum(abs(bbt_h(LTidx)));
    end
end

% construct new predictor matrix given Beta
X = [ones(n,1),bWb(:,act_comp)]; % n x (nact+1)

% OLS
tmp = (X'*X)\(X'* y); % (nact+1) x 1

alpha = tmp(1);
Lambda = zeros(K,1);
Lambda(act_comp) = tmp(2:end);

resid = y - X * tmp; % n x 1

LF(1) = sum(resid.^2)/2/n + gamma * bbt_sum' * abs(Lambda);

for iter = 2:fullit
    %% update Beta
    for h = 1:K
        if act_comp(h)            
            for u = 1:V
                comp_u = Lambda(h) * squeeze(W(u,:,:))'* Beta(:,h); % n x 1
                
                % residual if Beta(u,h)=0
                resid = resid + 2 * Beta(u,h).* comp_u; 
                
                A = 2 * sum(resid.* comp_u)/n;
                D = 4 * sum(comp_u.^2)/n;
                                
                if (D>0)
                    tmp = gamma * abs(Lambda(h)) * ( sum(abs(Beta(:,h))) - abs(Beta(u,h)) );
                    tmp = abs(A) - tmp;
                    
                    if (tmp>0)
                        Beta(u,h) = sign(A) * exp( log(tmp) - log(D) );
                        resid = resid - 2 * Beta(u,h).* comp_u;
                    else
                        Beta(u,h) = 0;
                    end                   
                else % D==0
                    Beta(u,h) = 0;
                end                       
            end
            % check empty
            if ( sum( Beta(:,h)~=0 ) < 2 )
                act_comp(h) = 0;
                Beta(:,h) = zeros(V,1);
                Lambda(h) = 0;
            end
        end
    end
    
    %% update Lambda
    for h = 1:K
        if act_comp(h)
            bbt_h = Beta(:,h) * Beta(:,h)'; % V x V
            bWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* W,1),2)); % n x 1
            bbt_sum(h) = sum(abs(bbt_h(LTidx)));
            
            % residual if Lambda(h)=0
            resid = resid + Lambda(h) * bWb(:,h); % n x 1
            
            C = sum(bWb(:,h).* resid)/n; 
            B = sum(bWb(:,h).^2)/n;
            
            if (B>0)
                tmp = abs(C) - gamma * bbt_sum(h);
                if (tmp>0)
                    Lambda(h) = sign(C) * exp(log(tmp) - log(B));
                    resid = resid - Lambda(h) * bWb(:,h);
                else
                    Lambda(h) = 0;
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                end
            else
                Lambda(h) = 0;
                act_comp(h) = 0;
                Beta(:,h) = zeros(V,1);
            end            
        end
    end
    
    %% update alpha
    resid = y - bWb * Lambda;
    alpha = sum(resid)/n;
    
    %% stopping rule
    resid = resid - alpha;
    LF(iter) = sum(resid.^2)/2/n + gamma * bbt_sum' * abs(Lambda);
    
    % disp(iter)
    
    if ( (LF(iter-1) - LF(iter)) < tol * abs(LF(iter-1)) || isnan(LF(iter)) )
        break
    end
end

%% only update nonzero parameters
if (iter==fullit) && (fullit < maxit)
    for iter = fullit+1 : maxit
        %% update Beta
        for h=1:K
            if act_comp(h)
                for u=1:V
                    if (Beta(u,h) ~= 0)
                        comp_u = Lambda(h) * squeeze(W(u,:,:))'* Beta(:,h); % n x 1
                        
                        % residual if Beta(u,h)=0
                        resid = resid + 2 * Beta(u,h).* comp_u;
                        
                        A = 2 * sum(resid.* comp_u)/n;
                        D = 4 * sum(comp_u.^2)/n;
                        
                        if (D>0)
                            tmp = gamma * abs(Lambda(h)) * ( sum(abs(Beta(:,h))) - abs(Beta(u,h)) );
                            tmp = abs(A) - tmp;
                            
                            if (tmp>0)
                                Beta(u,h) = sign(A) * exp( log(tmp) - log(D) );
                                resid = resid - 2 * Beta(u,h).* comp_u;
                            else
                                Beta(u,h) = 0;
                            end
                        else % D==0
                            Beta(u,h) = 0;
                        end
                    end
                end
                % check empty
                if ( sum( Beta(:,h)~=0 ) < 2 )
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                    Lambda(h) = 0;
                end
            end
        end
        
        %% update Lambda
        for h = 1:K
            if act_comp(h)
                bbt_h = Beta(:,h) * Beta(:,h)'; % V x V
                bWb(:,h) = squeeze(sum(sum(repmat(bbt_h,[1,1,n]).* W,1),2)); % n x 1
                bbt_sum(h) = sum(abs(bbt_h(LTidx)));
                
                % residual if Lambda(h)=0
                resid = resid + Lambda(h) * bWb(:,h); % n x 1
                
                C = sum(bWb(:,h).* resid)/n;
                B = sum(bWb(:,h).^2)/n;
                
                if (B>0)
                    tmp = abs(C) - gamma * bbt_sum(h);
                    if (tmp>0)
                        Lambda(h) = sign(C) * exp(log(tmp) - log(B));
                        resid = resid - Lambda(h) * bWb(:,h);
                    else
                        Lambda(h) = 0;
                        act_comp(h) = 0;
                        Beta(:,h) = zeros(V,1);
                    end
                else
                    Lambda(h) = 0;
                    act_comp(h) = 0;
                    Beta(:,h) = zeros(V,1);
                end
            end
        end
        
        %% update alpha
        resid = y - bWb * Lambda;
        alpha = sum(resid)/n;
        
        %% stopping rule
        resid = resid - alpha;
        LF(iter) = sum(resid.^2)/2/n + gamma * bbt_sum' * abs(Lambda);
        
        % disp(iter)
        
        if ( (LF(iter-1) - LF(iter)) < tol * abs(LF(iter-1)) || isnan(LF(iter)) )
            break
        end
    end
end

LF = LF(1:iter);