clear all;
clc;
close all;

region_ = 'region_y3';
setting_region = 'inc_2_10_th_1';
time_reigon = '7_AVG15_pad';
fileName = '_StartHour_7_AVG15_pad_theta1e+00_inc_2_10';
filename = fullfile(region_, setting_region, time_reigon, strcat('AllVar', fileName, '.mat'));
load(filename);

%%
beta0 = 1.0;
beta1 = 0.15;
tt0_array = readmatrix(fullfile(region_, strcat('Mar2May_2018_new_5-22_link_tt_0_minutes_', region_, '.csv')));
w_array = readmatrix(fullfile(region_, strcat('Mar2May_2018_new_5-22_link_capacity_', region_, '.csv')));
L_array = readmatrix(fullfile(region_, 'link_length_meter_region_y3_original.csv'))./1000;
F_speed = @(g, v, tt0, w, L) L*(tt0*(1 + 0.15*((g+v)/w)^4))^(-1);

m = size(A, 2);
n = sum(q);

S = zeros(m,n);
W = zeros(m,n);
H = zeros(m,n);
Omega = 10000; % Budget
u = zeros(m,1);
gamma = zeros(size(A,1),1);
beta = 0;
Q = zeros(m,n);
Q_best_loss = zeros(m,n);
Q_best_norm = zeros(m,n);

lambda1 = u;
lambda2 = zeros(n,1);
lambda3 = zeros(size(q));
lambda4 = zeros(size(gamma));
lambda5 = zeros(size(S));
%lambda6 = zeros(size(u));
lambda7 = 0;
%lambda8 = zeros(size(u));
lambda9 = zeros(size(W));
lambda10 = zeros(size(Q));

rho = 10; % Rho
lambda_H_list = [0.1, 1, 5, 9]; % Multiplier of H(H-1)
% lambda_H_list = [1, 5]; % Multiplier of H(H-1)
% if rho==lambda_H
%     msg = 'rho==lambda_H!!';
%     error(msg)
% end
MaxIter = 150001; % Max number of iteration for ADMM
% MaxIter = 101; % Max number of iteration for ADMM
MaxIterQ = 0;% Max number of iteration for ADMM-Q
min_loss = inf;
min_norm = inf;
min_norm2 = inf;
min_gap = inf;

tt0_array = readmatrix(fullfile(region_, strcat('Mar2May_2018_new_5-22_link_tt_0_minutes_', region_, '.csv')));
w_array = readmatrix(fullfile(region_, strcat('Mar2May_2018_new_5-22_link_capacity_', region_, '.csv')));
F_gamma = @(g, v, tt0, w) 0.15*tt0*((g+v)^5)/(w^4) + (g+v)*tt0;
gradient_gamma = @(g, v, tt0, w, l, a, r) 5*0.15*tt0*((g+v)^4)/(w^4) + tt0 - l - r*(a*u-g);
error_bisection = 1e-5; % Max number of iteration for ADMM-Q
min_bis = 0; % Min of domain
max_bis = 600; % Max of domain
no_obj = false;
if no_obj
    no_obj_str = 'noObj';
else
    no_obj_str = 'wObj';
end
TempS = (ones(n,n) + 2*eye(n))^(-1);
fprintf('Fro Norm of TempS: %.4f\n', norm(TempS,'fro'))
TempS2 = (ones(n,n) + 3*eye(n))^(-1);
fprintf('Fro Norm of TempS2: %.4f\n', norm(TempS2,'fro'))
Temptheta = eye(numel(c)) - 1/(1+norm(c)^2)* c*c';
TempW =  (eye(m) + ones(m,m))^(-1);
fprintf('Fro Norm of TempW: %.4f\n', norm(TempW,'fro'))
Tempu = (eye(m) + D'*D + A'*A + c*c')^(-1);
fprintf('Fro Norm of Tempu: %.4f\n', norm(Tempu,'fro'))

lagrangian = zeros(MaxIter,1);
ConstViolation = zeros(MaxIter,1);
ConstViolation_normalized = zeros(MaxIter,1);
total_travel_time_array = zeros(MaxIter,1);

fprintf('Norm of u: %.4f\n', norm(u))
fprintf('Fro Norm of W: %.4f\n', norm(W,'fro'))
fprintf('Fro Norm of H: %.4f\n', norm(H,'fro'))
fprintf('Fro Norm of S: %.4f\n', norm(S,'fro'))
fprintf('Norm of gamma: %.4f\n', norm(gamma))
fprintf('Norm of beta: %.4f\n', norm(beta))

counter_save = 1;

time_array = zeros(100, 7);
for iter = 1:(MaxIter+MaxIterQ)
    permutation_order = randi([0, 1]);
    if permutation_order==1
        % Block 1
        if iter<=MaxIter
            tic 
            S = (1/rho)*(-lambda1*ones(1,n) + lambda5 + lambda9 + rho * u * ones(1,n) + rho * H + rho*W)*TempS;
            time_array(iter, 1) = toc;
        else
            S = (1/rho)*(-lambda1*ones(1,n) + lambda5 + lambda9 + rho * u * ones(1,n) + rho * H + rho*W + lambda10 + rho*Q)*TempS2;
        end
        tic
        if no_obj
            F_gamma_total = 0;
            gamma = (1/rho) * lambda4 + A*u;
            for iter_gamma = 1:size(gamma, 1)
                tt0 = tt0_array(mod(iter_gamma, size(tt0_array, 1)) + 1, 3);
                w = w_array(mod(iter_gamma, size(w_array, 1)) + 1, 3);
                a = A(iter_gamma, :);
                l = lambda4(iter_gamma);
                v = v2(iter_gamma) + v3(iter_gamma) + v4(iter_gamma);
                F_gamma_total = F_gamma_total + F_gamma(gamma(iter_gamma, 1), v, tt0, w);
            end
        else
            F_gamma_total = 0;
            for iter_gamma = 1:size(gamma, 1)
                tt0 = tt0_array(mod(iter_gamma, size(tt0_array, 1)) + 1, 3);
                w = w_array(mod(iter_gamma, size(w_array, 1)) + 1, 3);
                a = A(iter_gamma, :);
                l = lambda4(iter_gamma);
                v = v2(iter_gamma) + v3(iter_gamma) + v4(iter_gamma); 
                gamma(iter_gamma) = bisectionMethodNew(gradient_gamma, v, min_bis, max_bis, error_bisection, tt0, w, l, a, rho);
            end
        end
        time_array(iter, 2) = toc;
        
        tic
        beta = (1/rho) * (-lambda7 - rho*c'*u+rho*Omega);
        if beta<0
            beta = 0;
        end
        time_array(iter, 3) = toc;

        
        % Block 2
        tic
        u = (1/rho)* Tempu * (lambda1 +rho*S*ones(n,1) - D'*lambda3 +  ...
            rho*D'*q - A'*lambda4 + rho*A'*gamma - c*lambda7-c*rho*(beta-Omega));
        time_array(iter, 4) = toc;
        
        tic
        W = (1/rho)*TempW*(-ones(m,1)*lambda2' - lambda9 + ...
            rho*ones(size(S)) + rho *S);
        time_array(iter, 5) = toc;
        
        if iter<=MaxIter/6
            lambda_H = lambda_H_list(1);
        elseif  (iter>MaxIter/30) && (2*MaxIter/30)
            lambda_H = lambda_H_list(2);
        elseif (2*iter>MaxIter/30) && (4*MaxIter/30)
            lambda_H = lambda_H_list(3);
        else
            lambda_H = lambda_H_list(4);
        end
        tic 
        H = (1/(rho-lambda_H))*(-lambda5 + rho*S - 0.5*lambda_H);
        
        if rho>lambda_H
            H(H<0) = 0;
            H(H>1) = 1;
        else
            H(H<=0.5) = 1;
            H(H>0.5) = 0;
        end
        time_array(iter, 6) = toc;
    else
        % Block 2
        u = (1/rho)* Tempu * (lambda1 +rho*S*ones(n,1) - D'*lambda3 +  ...
            rho*D'*q - A'*lambda4 + rho*A'*gamma - c*lambda7-c*rho*(beta-Omega));
        W = (1/rho)*TempW*(-ones(m,1)*lambda2' - lambda9 + ...
            rho*ones(size(S)) + rho *S);
        if iter<=MaxIter/6
            lambda_H = lambda_H_list(1);
        elseif  (iter>MaxIter/30) && (2*MaxIter/30)
            lambda_H = lambda_H_list(2);
        elseif (2*iter>MaxIter/30) && (4*MaxIter/30)
            lambda_H = lambda_H_list(3);
        else
            lambda_H = lambda_H_list(4);
        end
        H = (1/(rho-lambda_H))*(-lambda5 + rho*S - 0.5*lambda_H);
        if rho>lambda_H
            H(H<0) = 0;
            H(H>1) = 1;
        else
            H(H<=0.5) = 1;
            H(H>0.5) = 0;
        end
        
        % Block 1
        if iter<=MaxIter
            S = (1/rho)*(-lambda1*ones(1,n) + lambda5 + lambda9 + rho * u * ones(1,n) + rho * H + rho*W)*TempS;
        else
            S = (1/rho)*(-lambda1*ones(1,n) + lambda5 + lambda9 + rho * u * ones(1,n) + rho * H + rho*W + lambda10 + rho*Q)*TempS2;
        end
        if no_obj
            F_gamma_total = 0;
            gamma = (1/rho) * lambda4 + A*u;
            for iter_gamma = 1:size(gamma, 1)
                tt0 = tt0_array(mod(iter_gamma, size(tt0_array, 1)) + 1, 3);
                w = w_array(mod(iter_gamma, size(w_array, 1)) + 1, 3);
                a = A(iter_gamma, :);
                l = lambda4(iter_gamma);
                v = v2(iter_gamma) + v3(iter_gamma) + v4(iter_gamma);
                F_gamma_total = F_gamma_total + F_gamma(gamma(iter_gamma, 1), v, tt0, w);
            end
        else
            F_gamma_total = 0;
            for iter_gamma = 1:size(gamma, 1)
                tt0 = tt0_array(mod(iter_gamma, size(tt0_array, 1)) + 1, 3);
                w = w_array(mod(iter_gamma, size(w_array, 1)) + 1, 3);
                a = A(iter_gamma, :);
                l = lambda4(iter_gamma);
                v = v2(iter_gamma) + v3(iter_gamma) + v4(iter_gamma); 
                gamma(iter_gamma) = bisectionMethodNew(gradient_gamma, v, min_bis, max_bis, error_bisection, tt0, w, l, a, rho);
                F_gamma_total = F_gamma_total + F_gamma(gamma(iter_gamma, 1), v, tt0, w);
            end
        end
        total_travel_time_array(iter, 1) = F_gamma_total;
        beta = (1/rho) * (-lambda7 - rho*c'*u+rho*Omega);
        if beta<0
            beta = 0;
        end
%         beta = 0;
    end
    
    tic
    lambda1 = lambda1 + rho*(S*ones(n,1)-u);
    lambda2 = lambda2 + rho*(W'*ones(m,1)-ones(n,1));
    lambda3 = lambda3 + rho*(D*u -q);
    lambda4 = lambda4 + rho*(A*u-gamma);
    lambda5 = lambda5 + rho*(H-S);
    lambda7 = lambda7 + rho*(c'*u+beta-Omega);
    lambda9 = lambda9 + rho*(W-S);
    if iter>MaxIter
        lambda10 = lambda10 + rho*(Q-S);
        fprintf('Norm of lambda10: %.4f\n', norm(lambda10,'fro'))
    end
    time_array(iter, 7) = toc;
    
    %     if no_obj || iter<MaxIter/2
    if no_obj
        F_gamma_total = 0;
        for iter_gamma = 1:size(gamma, 1)
            tt0 = tt0_array(mod(iter_gamma, size(tt0_array, 1)) + 1, 3);
            w = w_array(mod(iter_gamma, size(w_array, 1)) + 1, 3);
            v = v2(iter_gamma) + v3(iter_gamma) + v4(iter_gamma); 
            F_gamma_total = F_gamma_total + F_gamma(gamma(iter_gamma, 1), v, tt0, w);
        end
    end
    lagrangian(iter) = F_gamma_total + lambda1'*(S*ones(n,1)-u)+...
        lambda2'*(W'*ones(m, 1) - ones(n,1)) + lambda3'*(D*u-q) + lambda4'*(A*u-gamma) + ...
        norm(lambda5'*(H-S), 'fro') + lambda7'*(c'*u+beta-Omega) + norm(lambda9'*(W-S), 'fro') + ...
        rho/2*norm(S*ones(n, 1) - u)^2 + rho/2*norm(H-S, 'fro')^2 + ...
        rho/2*norm(W'*ones(m, 1) - ones(n, 1))^2 + rho/2*norm(D*u - q)^2 + ...
        rho/2*norm(A*u - gamma)^2 + rho/2*(c'*u + beta - Omega)^2 + ...
        rho/2*norm(W-S, 'fro')^2;

    ConstViolation_normalized(iter) = norm(S*ones(n,1)-u)/(norm(S)*norm(ones(n,1)) + norm(u)) + ...
            norm(W'*ones(m,1)-ones(n,1))/(norm(W')*norm(ones(m,1)) + norm(ones(n,1))) + ...
            norm(D*u -q)/(norm(D)*norm(u) + norm(q)) + ...
            norm(H-S,'fro')/(norm(H, 'fro') + norm(S, 'fro')) + ...
            norm(A*u-gamma)/(norm(A)*norm(u) + norm(gamma)) +...
            norm(c'*u+beta-Omega)/(norm(c')*norm(u) + norm(beta) + norm(Omega)) + ...
            norm(W-S,'fro')/(norm(W, 'fro') + norm(S, 'fro'));
    ConstViolation(iter) = norm(S*ones(n,1)-u)+norm(W'*ones(m,1)-ones(n,1))+norm(D*u -q)+ norm(H-S,'fro') + norm(A*u-gamma)+norm(c'*u+beta-Omega) + norm(W-S,'fro');

    if iter>MaxIter
        ConstViolation_normalized(iter) = ConstViolation_normalized(iter) + ...
            norm(Q-S,'fro')/(norm(Q, 'fro') + norm(S, 'fro'));
        ConstViolation(iter) = ConstViolation(iter) + norm(Q-S,'fro');
    end
    
    if mod(iter, 25)==0
        iter
        fprintf('Norm of u: %.10f\n', norm(u))
        fprintf('Fro Norm of W: %.10f\n', norm(W,'fro'))
        fprintf('Fro Norm of H: %.10f\n', norm(H,'fro'))
        fprintf('Fro Norm of S: %.10f\n', norm(S,'fro'))
        fprintf('Norm of gamma: %.10f\n', norm(gamma))
        fprintf('Norm of beta: %.10f\n', norm(beta))
        
        fprintf('Norm of Gap1 (normalized): %.10f\n', norm(S*ones(n,1)-u)/(norm(S)*norm(ones(n,1) + norm(u))))
        fprintf('Fro Norm of Gap2 (normalized): %.10f\n', norm(W'*ones(m,1)-ones(n,1))/(norm(W')*norm(ones(m,1)) + norm(ones(n,1))))
        fprintf('Fro Norm of Gap3 (normalized): %.10f\n', norm(D*u -q)/(norm(D)*norm(u) + norm(q)))
        fprintf('Fro Norm of Gap4 (normalized): %.10f\n', norm(H-S,'fro')/(norm(H, 'fro') + norm(S, 'fro')))
        fprintf('Norm of Gap5 (normalized): %.10f\n', norm(A*u-gamma)/(norm(A)*norm(u) + norm(gamma)))
        fprintf('Norm of Gap6 (normalized): %.10f\n', norm(c'*u+beta-Omega)/(norm(c')*norm(u) + norm(beta) + norm(Omega)))
        fprintf('Norm of Gap7 (normalized): %.10f\n', norm(W-S,'fro')/(norm(W, 'fro') + norm(S, 'fro')))
        if iter>MaxIter
            fprintf('Norm of Gap8 (normalized): %.10f\n', norm(Q-S,'fro')/(norm(Q, 'fro') + norm(S, 'fro')))
        end
        fprintf('Gap without normalization: %.10f\n', ConstViolation(iter))
        fprintf('Gap with normalization: %.10f\n', ConstViolation_normalized(iter))
        fprintf('Lagrangian value: %.10f\n', lagrangian(iter))
        fprintf('Total travel time: %.10f\n', F_gamma_total)
        S2 = S;
        S2(S<0)=1;
        S2(S>=0)=0;
        percNegS = (size(S2,1)*size(S2,2) - sum(sum(S2)))/(size(S2,1)*size(S2,2))*100;
        fprintf('Percentage of positive values of S: %.10f\n', percNegS)
        u2 = u;
        u2(u<0)=1;
        u2(u>=0)=0;
        percNegu = (size(u2,1) - sum(u2))/(size(u2,1))*100;
        fprintf('Percentage of positive values of u: %.10f\n', percNegu)
        gamma2 = gamma;
        gamma2(gamma<0)=1;
        gamma2(gamma>=0)=0;
        percNeggamma = (size(gamma2,1) - sum(gamma2))/(size(gamma2,1))*100;
        fprintf('Percentage of positive values of gamma: %.10f\n', percNeggamma)
    end
    
    if iter>MaxIter-10
        iter
        fprintf('\n\n\nNorm of u: %.10f\n', norm(u))
        fprintf('Fro Norm of W: %.10f\n', norm(W,'fro'))
        fprintf('Fro Norm of H: %.10f\n', norm(H,'fro'))
        fprintf('Fro Norm of S: %.10f\n', norm(S,'fro'))
        fprintf('Norm of gamma: %.10f\n', norm(gamma))
        fprintf('Norm of beta: %.10f\n', norm(beta))
        fprintf('Norm of Q: %.4f\n', norm(Q,'fro'))
        
        fprintf('Norm of Gap1 (normalized): %.10f\n', norm(S*ones(n,1)-u)/(norm(S)*norm(ones(n,1) + norm(u))))
        fprintf('Fro Norm of Gap2 (normalized): %.10f\n', norm(W'*ones(m,1)-ones(n,1))/(norm(W')*norm(ones(m,1)) + norm(ones(n,1))))
        fprintf('Fro Norm of Gap3 (normalized): %.10f\n', norm(D*u -q)/(norm(D)*norm(u) + norm(q)))
        fprintf('Fro Norm of Gap4 (normalized): %.10f\n', norm(H-S,'fro')/(norm(H, 'fro') + norm(S, 'fro')))
        fprintf('Norm of Gap5 (normalized): %.10f\n', norm(A*u-gamma)/(norm(A)*norm(u) + norm(gamma)))
        fprintf('Norm of Gap6 (normalized): %.10f\n', norm(c'*u+beta-Omega)/(norm(c')*norm(u) + norm(beta) + norm(Omega)))
        fprintf('Norm of Gap7 (normalized): %.10f\n', norm(W-S,'fro')/(norm(W, 'fro') + norm(S, 'fro')))
        if iter>MaxIter
            fprintf('Norm of Gap8 (normalized): %.10f\n', norm(Q-S,'fro')/(norm(Q, 'fro') + norm(S, 'fro')))
        end
        fprintf('Gap without normalization: %.10f\n', ConstViolation(iter))
        fprintf('Gap with normalization: %.10f\n', ConstViolation_normalized(iter))
        fprintf('Lagrangian value: %.10f\n', lagrangian(iter))
        fprintf('Total travel time: %.10f\n', F_gamma_total)
        S2 = S;
        S2(S<0)=1;
        S2(S>=0)=0;
        percNegS = (size(S2,1)*size(S2,2) - sum(sum(S2)))/(size(S2,1)*size(S2,2))*100;
        fprintf('Percentage of positive values of S: %.10f\n', percNegS)
        u2 = u;
        u2(u<0)=1;
        u2(u>=0)=0;
        percNegu = (size(u2,1) - sum(u2))/(size(u2,1))*100;
        fprintf('Percentage of positive values of u: %.10f\n', percNegu)
        gamma2 = gamma;
        gamma2(gamma<0)=1;
        gamma2(gamma>=0)=0;
        percNeggamma = (size(gamma2,1) - sum(gamma2))/(size(gamma2,1))*100;
        fprintf('Percentage of positive values of gamma: %.10f\n', percNeggamma)
        size_S = size(S);
        size_S1 = size(S, 1);
        size_S2 = size(S, 2);
        
        cvx_solver Gurobi
        cvx_solver_settings( 'MIPGap', .01 );
        cvx_begin
        variable S_sol(size_S) binary;
        minimize(sum(abs(S_sol * ones(size(S, 2), 1) - u )))
        subject to
        S_sol' * ones(size(S, 1), 1) == ones(size(S, 2), 1);
        D*S_sol*ones(size(S, 2), 1) == q;
        c'*S_sol*ones(size(S, 2), 1) <= Omega;
        cvx_end
        
        S_sol_full = full(S_sol);
        
        % Check the total travel time
        tt = @(x, v, tt0, w) (x+v)*tt0 + 0.15*tt0*((x+v)^5)/(w^4);
        ones_S2 = ones(size(S_sol_full, 2), 1);
        
        gamma_sol = A*S_sol_full*ones_S2;
        tt_obj = zeros(size(gamma));
        for iter_gamma = 1:size(gamma_sol, 1)
            tt0 = tt0_array(mod(iter_gamma, size(tt0_array, 1)) + 1, 3);
            w = w_array(mod(iter_gamma, size(w_array, 1)) + 1, 3);
            v = v2(iter_gamma) + v3(iter_gamma) + v4(iter_gamma); 

            tt_obj(iter_gamma) = tt(gamma_sol(iter_gamma, 1), v, tt0, w);
        end
        tt_obj_total = sum(tt_obj)/60
        
        ones_S1 = ones(size(S_sol_full, 1), 1);
        ones_S2 = ones(size(S_sol_full, 2), 1);
        const1_gap = S_sol_full'*ones_S1 - ones_S2;
        const1_gap_positive =  const1_gap(const1_gap~=0)
        
        const2_gap = c'*S_sol_full*ones_S2 - Omega
        
        const3_gap = D*S_sol_full*ones_S2 - q;
        const3_gap_positive = const3_gap(const3_gap~=0)
        counter_save = counter_save + 1;
        if lambda_H~=0
            %     outputFolder = fullfile(region_, setting_region, time_reigon, strcat('6AM_regularized_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega), '_lambdaH', num2str(lambda_H)));
            outputFolder = fullfile(region_, setting_region, time_reigon, strcat('7AM_regularized_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega), '_lambdaH', '_save', num2str(counter_save), '_randomPermutation'));
        else
            %     outputFolder = fullfile(region_, setting_region, time_reigon, strcat('6AM_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega)));
            outputFolder = fullfile(region_, setting_region, time_reigon, strcat('7AM_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega), '_save', num2str(counter_save), '_randomPermutation'));
        end
        mkdir(outputFolder);
        filenameOutput = fullfile(outputFolder, 'AllVarOutput.mat');
        save(filenameOutput)
    end

iter
if iter==100
    iter % #########################################
    fprintf('Computation time of S: %.6f\n', sum(time_array(:, 1))/100)
    fprintf('Computation time of gamma: %.6f\n', sum(time_array(:, 2))/100)
    fprintf('Computation time of beta: %.6f\n', sum(time_array(:, 3))/100)
    fprintf('Computation time of u: %.6f\n', sum(time_array(:, 4))/100)
    fprintf('Computation time of W: %.6f\n', sum(time_array(:, 5))/100)
    fprintf('Computation time of H: %.6f\n', sum(time_array(:, 6))/100)
    fprintf('Computation time of dual variables: %.6f\n', sum(time_array(:, 7))/100)

    iter
end
fprintf('\n\n\n')
end


%%
tic
if lambda_H~=0
    %     outputFolder = fullfile(region_, setting_region, time_reigon, strcat('6AM_regularized_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega), '_lambdaH', num2str(lambda_H)));
    outputFolder = fullfile(region_, setting_region, time_reigon, strcat('7AM_regularized_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega), '_lambdaH_randomPermutation'));
else
    %     outputFolder = fullfile(region_, setting_region, time_reigon, strcat('6AM_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega)));
    outputFolder = fullfile(region_, setting_region, time_reigon, strcat('7AM_', no_obj_str, '_rho', num2str(rho), '_MaxIter', num2str(MaxIter), '_MaxIterQ', num2str(MaxIterQ), '_Omega', num2str(Omega), '_randomPermutation'));
end
mkdir(outputFolder);
filenameOutput = fullfile(outputFolder, 'AllVarOutput.mat');
save(filenameOutput)

plot(ConstViolation_normalized)
plot(total_travel_time_array)
toc
