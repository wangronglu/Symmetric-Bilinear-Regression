function [alpha_final, B_final, Lambda_final, F_final] = SymBilinear(W,y,K,gamma,fullit,maxit,tol,B0,Replicates)

% SymBilinear fits the symmetric rank-K Kruskal tensor regression with
% L1 constraint
% Loss function = RSS/(2n) + L1 penalty
% L1 penalty = gamma * sum of abs value of coefficients
%               in each matrix (lambda_h * beta_h * beta_h^T)
% use coordinate descent to update each entry in \beta_h 
% y_i = \alpha + \sum_{h=1}^K lambda_h * t(\beta_h) * W_i * \beta_h + \epsilon_i
%
% Input:
%   W: 3D array variates with dim(W) = [V,V,n]; W(:,:,i) is symmetric with
%       zero diagonal
%   y: n-by-1 response vector
%   K: rank of Kruskal tensor regression
%   gamma: penalty parameter for L1 norm, eg. gamma = 1;
%
% Optional input:
%   fullit: number of iterations that cycle all variables; after that only
%       update active variables.(should smaller than maxit, default 100)
%   maxit: maximum iterations (default 10000)
%   tol: relative change tolerance in objective function. default tol=1e-5 
%   B0: initial B
%       * do not initialize B at zero vectors
%   Replicates: number of initial points to try (default 5)
%
% Output:
%   alpha_final: (scalar) intercept of regression
%   B_final: a VxK matrix containing the tensor regression coefficient
%            vectors
%   Lambda_final: a Kx1 vector 
%   F_final: evolution of objective function across iterations 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(fullit)
    fullit = 100;
end

if isempty(maxit)
    maxit = 10000;
end

if isempty(tol)
    tol = 1e-5;
end

ini_flag = 1;
if isempty(B0)
    ini_flag = 0;
end

if isempty(Replicates)
    Replicates = 5;
end

V = size(W,1); 
n = length(y);
% lower triangular index
LTidx = tril(true(V),-1);

% intermediate variables
M = zeros(V,V,V);
for u=1:V
    M_u = zeros(V,V);
    for i=1:n
        M_u = M_u + W(:,u,i) * W(u,:,i);
    end
    M(:,:,u) = M_u;
end

resid_rmh = zeros(n,1);

%% If B0 is given
if ini_flag
    % B_all = zeros(V,K,maxit);
    % alpha_all = zeros(maxit,1);
    % Lambda_all = zeros(K,maxit);
    
    %% Initialize alpha and Lambda
    B = B0;
    
    % B_all(:,:,1) = B; 
    
    % detect degenerate components
    active_Lambda = (B ~= 0);
    active_Lambda = (sum(active_Lambda) > 1);
    
    nact = sum(active_Lambda);
    if (nact<1)
        B = zeros(V,K);
        Lambda = zeros(K,1);
        alpha = mean(y);
                              
        % alpha_all(1) = alpha;        
        % Lambda_all(:,1) = Lambda;
        
        % objective function
        resid = y - alpha;
        F1 = zeros(maxit,1);        
        F1(1) = (resid'*resid)/2/n;
    else    
        B(:, ~active_Lambda) = zeros(V,K-nact);
        
        % initialize alpha and Lambda ------------------------
        % construct new predictor matrix
        X = [ones(n,1),zeros(n,nact)];
        for i=1:n
            X(i,2:end) = diag(B(:,active_Lambda)'* W(:,:,i) * B(:,active_Lambda))';
        end
        % OLS + ridge penalty for robust estimation
        ctmp = (X'*X + 1e-6 *eye(nact+1))\(X' * y);
        alpha = ctmp(1);
        
        % alpha_all(1) = alpha;
        
        Lambda = zeros(K,1);
        Lambda(active_Lambda) = ctmp(2:end);
        
        % Lambda_all(:,1) = Lambda;
        
        % store residuals
        resid = y - X * ctmp;
        
        % update active_components
        active_Lambda = (Lambda ~= 0);
        nact = sum(active_Lambda);
        B(:, ~active_Lambda) = zeros(V,K-nact);

        % objective function ----
        F1 = zeros(maxit,1);
        % penalty
        sum_abs_bbt = zeros(K,1);
        for h = 1:K
            bbt = B(:,h) * B(:,h)';
            sum_abs_bbt(h) = sum(abs(bbt(LTidx)));
        end
        
        F1(1) = (resid'*resid)/2/n + gamma * sum_abs_bbt' * abs(Lambda);
 
    end
    
    %% Iteration: cycle all the variables
    for iter = 2 : fullit
        %% update B
        for h=1:K
            if ~ active_Lambda(h)
                continue
            else 
                % residual excluding component h
                act_Lambda_rmh = active_Lambda;
                act_Lambda_rmh(h) = 0;
                for i=1:n
                    resid_rmh(i) = y(i) - alpha - ...
                        sum(Lambda(act_Lambda_rmh)'* ...
                        diag(B(:,act_Lambda_rmh)'* W(:,:,i) * B(:,act_Lambda_rmh)));
                end
                
                % update each entry in B(:,h)
                for u=1:V
                    A = 0;
                    for i=1:n
                        bWb_i = B(:,h)' * W(:,:,i) * B(:,h);
                        c_iu = W(u,:,i) * B(:,h);
                        m_iu = bWb_i - 2 * c_iu * B(u,h);
                        A = A + (resid_rmh(i) - Lambda(h) * m_iu) * c_iu;
                    end
                    A = 2 * Lambda(h) * A / n;
                    D = 4 * Lambda(h)^2 / n * B(:,h)' * M(:,:,u) * B(:,h);
                    if (D==0)
                        B(:,h) = zeros(V,1);
                        active_Lambda(h) = 0;
                        Lambda(h) = 0;
                        break
                    else
                        pen_B_hu = gamma * abs(Lambda(h))*( sum(abs(B(:,h))) - abs(B(u,h)) );
                        tmp_diff = abs(A) - pen_B_hu;
                        if (tmp_diff > 0)
                            B(u,h) = sign(A) * exp( log(tmp_diff) - log(D) );
                        else
                            B(u,h) = 0;
                        end
                    end
                end
            end
        end
        
        % B_all(:,:,iter) = B;
        
        %% update Lambda and alpha
        nact = sum(active_Lambda);
        if (nact<1)
            alpha = mean(y); % Lambda is a zero vector now
            
            % alpha_all(iter) = alpha;
            % Lambda_all(:,iter) = Lambda;
            
            % objective function
            resid = y - alpha;
            F1(iter) = (resid'*resid)/2/n;
        else
            % construct new predictor matrix
            X = zeros(n,K);
            for i=1:n
                X(i,:) = diag(B'* W(:,:,i) * B)';
            end
            
            pen_Lam = zeros(K,1);
            % update Lambda -----------------------------------------------
            for h = 1:K
                if ~ active_Lambda(h)
                    continue
                else
                    % residual excluding component h
                    Lambda_rmh = Lambda;
                    Lambda_rmh(h) = 0;
                    resid_rmh = y - alpha - X * Lambda_rmh;

                    G_h = X(:,h)'* resid_rmh / n;
                    H_h = X(:,h)'* X(:,h) / n;
                    if (H_h == 0)
                        Lambda(h) = 0;
                        active_Lambda(h) = 0;
                        B(:,h) = zeros(V,1);
                    else
                        bbt = B(:,h) * B(:,h)';
                        pen_Lam(h) = gamma * sum(abs(bbt(LTidx)));
                        tmp_diff = abs(G_h) - pen_Lam(h);
                        if (tmp_diff > 0)
                            Lambda(h) = sign(G_h) * exp( log(tmp_diff) - log(H_h) );
                        else
                            Lambda(h) = 0;
                            active_Lambda(h) = 0;
                            B(:,h) = zeros(V,1);
                        end
                    end
                end
            end
            
            % Lambda_all(:,iter) = Lambda;
            
            % update alpha -------------------------
            resid_rmalp = y - X * Lambda;
            alpha = sum(resid_rmalp)/n ;
            
            % alpha_all(iter) = alpha;
 
            % objective function
            resid = resid_rmalp - alpha;
           
            F1(iter) = (resid'*resid)/2/n + pen_Lam' * abs(Lambda);
            
        end
        %% stopping rule
        if ( F1(iter-1) - F1(iter) < tol * F1(iter-1) )
            break
        end
    end
    
    if (iter == fullit)
        %% Iterations: only active set of B
        for iter = fullit+1 : maxit
            %% update B
            for h=1:K
                if ~ active_Lambda(h)
                    continue
                else
                    % residual excluding component h
                    act_Lambda_rmh = active_Lambda;
                    act_Lambda_rmh(h) = 0;
                    for i=1:n
                        resid_rmh(i) = y(i) - alpha - ...
                            sum(Lambda(act_Lambda_rmh)'* ...
                            diag(B(:,act_Lambda_rmh)'* W(:,:,i) * B(:,act_Lambda_rmh)));
                    end
                    
                    % only update nonzero entry in B(:,h)
                    for u=1:V
                        if ( B(u,h) == 0 )
                            continue
                        end
                        A = 0;
                        for i=1:n
                            bWb_i = B(:,h)' * W(:,:,i) * B(:,h);
                            c_iu = W(u,:,i) * B(:,h);
                            m_iu = bWb_i - 2 * c_iu * B(u,h);
                            A = A + (resid_rmh(i) - Lambda(h) * m_iu) * c_iu;
                        end
                        A = 2 * Lambda(h) * A / n;
                        D = 4 * Lambda(h)^2 / n * B(:,h)' * M(:,:,u) * B(:,h);
                        if (D==0)
                            B(:,h) = zeros(V,1);
                            active_Lambda(h) = 0;
                            Lambda(h) = 0;
                            break
                        else
                            pen_B_hu = gamma * abs(Lambda(h))*( sum(abs(B(:,h))) - abs(B(u,h)) );
                            tmp_diff = abs(A) - pen_B_hu;
                            if (tmp_diff > 0)
                                B(u,h) = sign(A) * exp( log(tmp_diff) - log(D) );
                            else
                                B(u,h) = 0;
                            end
                        end
                    end
                end
            end
            
            % B_all(:,:,iter) = B;
            
            %% update Lambda and alpha
            nact = sum(active_Lambda);
            if (nact<1)
                alpha = mean(y); % Lambda is a zero vector now
                
                % alpha_all(iter) = alpha;
                % Lambda_all(:,iter) = Lambda;
                
                % objective function
                resid = y - alpha;
                F1(iter) = (resid'*resid)/2/n;
            else
                % construct new predictor matrix
                X = zeros(n,K);
                for i=1:n
                    X(i,:) = diag(B'* W(:,:,i) * B)';
                end
                
                pen_Lam = zeros(K,1);
                % update Lambda -----------------------------------------------
                for h = 1:K
                    if ~ active_Lambda(h)
                        continue
                    else
                        % residual excluding component h
                        Lambda_rmh = Lambda;
                        Lambda_rmh(h) = 0;
                        resid_rmh = y - alpha - X * Lambda_rmh;
                        
                        G_h = X(:,h)'* resid_rmh / n;
                        H_h = X(:,h)'* X(:,h) / n;
                        if (H_h == 0)
                            Lambda(h) = 0;
                            active_Lambda(h) = 0;
                            B(:,h) = zeros(V,1);
                        else
                            bbt = B(:,h) * B(:,h)';
                            pen_Lam(h) = gamma * sum(abs(bbt(LTidx)));
                            tmp_diff = abs(G_h) - pen_Lam(h);
                            if (tmp_diff > 0)
                                Lambda(h) = sign(G_h) * exp( log(tmp_diff) - log(H_h) );
                            else
                                Lambda(h) = 0;
                                active_Lambda(h) = 0;
                                B(:,h) = zeros(V,1);
                            end
                        end
                    end
                end
                
                % Lambda_all(:,iter) = Lambda;
                
                % update alpha -------------------------
                resid_rmalp = y - X * Lambda;
                alpha = sum(resid_rmalp)/n ;
                
                % alpha_all(iter) = alpha;
                
                % objective function
                resid = resid_rmalp - alpha;
                
                F1(iter) = (resid'*resid)/2/n + pen_Lam' * abs(Lambda);
                
            end
            %% stopping rule
            if ( F1(iter-1) - F1(iter) < tol * F1(iter-1) )
                break
            end
            
        end
    end
    
    alpha_final = alpha;
    B_final = B;
    Lambda_final = Lambda;
    F_final = F1(1:iter);
    Fmin = F1(iter);
else
    % If no initialization for B
    Fmin = inf;
end
    
%% Replicates
for rep = 1:Replicates
    disp(rep)
    %% Initialize B, alpha and Lambda
    % do not initialize B at zero vectors
    B = 1 - 2 * rand(V,K);
    
    % detect degenerate components
    active_Lambda = (B ~= 0);
    active_Lambda = (sum(active_Lambda) > 1);
    
    nact = sum(active_Lambda);
    if (nact<1)
        B = zeros(V,K);
        Lambda = zeros(K,1);
        alpha = mean(y);
        
        % objective function
        resid = y - alpha;
        F1 = zeros(maxit,1);
        F1(1) = (resid'*resid)/2/n;
    else
        B(:, ~active_Lambda) = zeros(V,K-nact);
        
        % initialize alpha and Lambda ------------------------
        % construct new predictor matrix
        X = [ones(n,1),zeros(n,nact)];
        for i=1:n
            X(i,2:end) = diag(B(:,active_Lambda)'* W(:,:,i) * B(:,active_Lambda))';
        end
        % OLS + ridge penalty for robust estimation
        ctmp = (X'*X + 1e-6 *eye(nact+1))\(X' * y);
        alpha = ctmp(1);
        
        Lambda = zeros(K,1);
        Lambda(active_Lambda) = ctmp(2:end);
        
        % store residuals
        resid = y - X * ctmp;
        
        % update active_components
        active_Lambda = (Lambda ~= 0);
        nact = sum(active_Lambda);
        B(:, ~active_Lambda) = zeros(V,K-nact);
        
        % objective function ----
        F1 = zeros(maxit,1);
        % penalty
        sum_abs_bbt = zeros(K,1);
        for h = 1:K
            bbt = B(:,h) * B(:,h)';
            sum_abs_bbt(h) = sum(abs(bbt(LTidx)));
        end
        
        F1(1) = (resid'*resid)/2/n + gamma * sum_abs_bbt' * abs(Lambda);
    end
    
    %% Iteration: cycle all the variables
    for iter = 2 : fullit
        %% update B
        for h=1:K
            if ~ active_Lambda(h)
                continue
            else
                % residual excluding component h
                act_Lambda_rmh = active_Lambda;
                act_Lambda_rmh(h) = 0;
                for i=1:n
                    resid_rmh(i) = y(i) - alpha - ...
                        sum(Lambda(act_Lambda_rmh)'* ...
                        diag(B(:,act_Lambda_rmh)'* W(:,:,i) * B(:,act_Lambda_rmh)));
                end
                
                % update each entry in B(:,h)
                for u=1:V
                    A = 0;
                    for i=1:n
                        bWb_i = B(:,h)' * W(:,:,i) * B(:,h);
                        c_iu = W(u,:,i) * B(:,h);
                        m_iu = bWb_i - 2 * c_iu * B(u,h);
                        A = A + (resid_rmh(i) - Lambda(h) * m_iu) * c_iu;
                    end
                    A = 2 * Lambda(h) * A / n;
                    D = 4 * Lambda(h)^2 / n * B(:,h)' * M(:,:,u) * B(:,h);
                    if (D==0)
                        B(:,h) = zeros(V,1);
                        active_Lambda(h) = 0;
                        Lambda(h) = 0;
                        break
                    else
                        pen_B_hu = gamma * abs(Lambda(h))*( sum(abs(B(:,h))) - abs(B(u,h)) );
                        tmp_diff = abs(A) - pen_B_hu;
                        if (tmp_diff > 0)
                            B(u,h) = sign(A) * exp( log(tmp_diff) - log(D) );
                        else
                            B(u,h) = 0;
                        end
                    end
                end
            end
        end
        
        %% update Lambda and alpha
        nact = sum(active_Lambda);
        if (nact<1)
            alpha = mean(y); % Lambda is a zero vector now
            
            % objective function
            resid = y - alpha;
            F1(iter) = (resid'*resid)/2/n;
        else
            % construct new predictor matrix
            X = zeros(n,K);
            for i=1:n
                X(i,:) = diag(B'* W(:,:,i) * B)';
            end
            
            pen_Lam = zeros(K,1);
            % update Lambda -----------------------------------------------
            for h = 1:K
                if ~ active_Lambda(h)
                    continue
                else
                    % residual excluding component h
                    Lambda_rmh = Lambda;
                    Lambda_rmh(h) = 0;
                    resid_rmh = y - alpha - X * Lambda_rmh;
                    
                    G_h = X(:,h)'* resid_rmh / n;
                    H_h = X(:,h)'* X(:,h) / n;
                    if (H_h == 0)
                        Lambda(h) = 0;
                        active_Lambda(h) = 0;
                        B(:,h) = zeros(V,1);
                    else
                        bbt = B(:,h) * B(:,h)';
                        pen_Lam(h) = gamma * sum(abs(bbt(LTidx)));
                        tmp_diff = abs(G_h) - pen_Lam(h);
                        if (tmp_diff > 0)
                            Lambda(h) = sign(G_h) * exp( log(tmp_diff) - log(H_h) );
                        else
                            Lambda(h) = 0;
                            active_Lambda(h) = 0;
                            B(:,h) = zeros(V,1);
                        end
                    end
                end
            end
            
            % update alpha -------------------------
            resid_rmalp = y - X * Lambda;
            alpha = sum(resid_rmalp)/n ;
            
            % objective function
            resid = resid_rmalp - alpha;
            F1(iter) = (resid'*resid)/2/n + pen_Lam' * abs(Lambda);
            
        end
        %% stopping rule
        if ( F1(iter-1) - F1(iter) < tol * F1(iter-1) )
            break
        end
    end
    
    if (iter == fullit)
        %% Iterations: only active set of B
        for iter = fullit+1 : maxit
            %% update B
            for h=1:K
                if ~ active_Lambda(h)
                    continue
                else
                    % residual excluding component h
                    act_Lambda_rmh = active_Lambda;
                    act_Lambda_rmh(h) = 0;
                    for i=1:n
                        resid_rmh(i) = y(i) - alpha - ...
                            sum(Lambda(act_Lambda_rmh)'* ...
                            diag(B(:,act_Lambda_rmh)'* W(:,:,i) * B(:,act_Lambda_rmh)));
                    end
                    
                    % only update nonzero entry in B(:,h)
                    for u=1:V
                        if ( B(u,h) == 0 )
                            continue
                        end
                        A = 0;
                        for i=1:n
                            bWb_i = B(:,h)' * W(:,:,i) * B(:,h);
                            c_iu = W(u,:,i) * B(:,h);
                            m_iu = bWb_i - 2 * c_iu * B(u,h);
                            A = A + (resid_rmh(i) - Lambda(h) * m_iu) * c_iu;
                        end
                        A = 2 * Lambda(h) * A / n;
                        D = 4 * Lambda(h)^2 / n * B(:,h)' * M(:,:,u) * B(:,h);
                        if (D==0)
                            B(:,h) = zeros(V,1);
                            active_Lambda(h) = 0;
                            Lambda(h) = 0;
                            break
                        else
                            pen_B_hu = gamma * abs(Lambda(h))*( sum(abs(B(:,h))) - abs(B(u,h)) );
                            tmp_diff = abs(A) - pen_B_hu;
                            if (tmp_diff > 0)
                                B(u,h) = sign(A) * exp( log(tmp_diff) - log(D) );
                            else
                                B(u,h) = 0;
                            end
                        end
                    end
                end
            end
            
            %% update Lambda and alpha
            nact = sum(active_Lambda);
            if (nact<1)
                alpha = mean(y); % Lambda is a zero vector now

                % objective function
                resid = y - alpha;
                F1(iter) = (resid'*resid)/2/n;
            else
                % construct new predictor matrix
                X = zeros(n,K);
                for i=1:n
                    X(i,:) = diag(B'* W(:,:,i) * B)';
                end
                
                pen_Lam = zeros(K,1);
                % update Lambda -----------------------------------------------
                for h = 1:K
                    if ~ active_Lambda(h)
                        continue
                    else
                        % residual excluding component h
                        Lambda_rmh = Lambda;
                        Lambda_rmh(h) = 0;
                        resid_rmh = y - alpha - X * Lambda_rmh;
                        
                        G_h = X(:,h)'* resid_rmh / n;
                        H_h = X(:,h)'* X(:,h) / n;
                        if (H_h == 0)
                            Lambda(h) = 0;
                            active_Lambda(h) = 0;
                            B(:,h) = zeros(V,1);
                        else
                            bbt = B(:,h) * B(:,h)';
                            pen_Lam(h) = gamma * sum(abs(bbt(LTidx)));
                            tmp_diff = abs(G_h) - pen_Lam(h);
                            if (tmp_diff > 0)
                                Lambda(h) = sign(G_h) * exp( log(tmp_diff) - log(H_h) );
                            else
                                Lambda(h) = 0;
                                active_Lambda(h) = 0;
                                B(:,h) = zeros(V,1);
                            end
                        end
                    end
                end

                % update alpha -------------------------
                resid_rmalp = y - X * Lambda;
                alpha = sum(resid_rmalp)/n ;

                % objective function
                resid = resid_rmalp - alpha;
                F1(iter) = (resid'*resid)/2/n + pen_Lam' * abs(Lambda);
                
            end
            %% stopping rule
            if ( F1(iter-1) - F1(iter) < tol * F1(iter-1) )
                break
            end
            
        end
    end
    
    % record if it has smaller objective function
    if F1(iter) < Fmin
        Fmin = F1(iter);
        F_final = F1(1:iter);
        alpha_final = alpha;
        B_final = B;
        Lambda_final = Lambda;
    end
    
end
