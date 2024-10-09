function BLP_results = optimize_model_SOL(alpha_income_init)

    % Checks if alpha_income_init was provided
    if nargin < 1
        error('Initial alpha_income value must be provided.');
    end

    clear global;
    clear data; 
    global invA ns x1 x2 s_jt IV vfull dfull theta1 theti thetj cdid cdindex old_theta2 delta_hat ind_sh

    product_data = readtable('product_data.csv');
    agent_data = readtable('agent_data.csv');

    ns = 20;
    nmkt = 94;
    nbrn = 24;
    n_inst = 20;

    cdid = kron((1:nmkt)', ones(nbrn,1));
    cdindex = (nbrn:nbrn:nbrn*nmkt)';

    IV = product_data{:, 6:26};

    v = reshape(agent_data.nodes0, 20, []).';
    demogr = reshape(agent_data.income, 20, []).';

    s_jt = product_data.shares;
    x1 = [ones(nmkt*nbrn, 1), product_data.prices];
    x2 = [ones(nmkt*nbrn, 1), product_data.prices, product_data.sugar];

    theta1 = zeros(size(x1,2),1);

    theta2_guess = [0                      4.5;
                    1        alpha_income_init;
                    0                    0.15];

    [theti, thetj, theta2] = find(theta2_guess);
    theti = double(theti);
    thetj = double(thetj);
    theta2 = double(theta2);
    
    temp = cumsum(s_jt);
    sum1 = temp(cdindex,:);
    sum1(2:size(sum1,1),:) = diff(sum1);
    outshr = 1.0 - sum1(cdid,:);

    delta = log(s_jt) - log(outshr);
    invA = inv(IV'*IV);
    theta1_hat = inv(x1'*IV*invA*IV'*x1)*x1'*IV*invA*IV'*delta;
    delta_hat = x1*theta1_hat;
    old_theta2 = zeros(size(theta2));
    delta_hat = exp(delta_hat);

    vfull = v(cdid,:);
    dfull = demogr(cdid,:);

    %options = optimset('GradObj','off','MaxIter',200,'Display','iter','TolFun',0.1,'TolX',0.01);
    options = optimset('GradObj', 'off', 'MaxIter', 1e20, 'MaxFunEvals', 1e20, 'Display', 'iter', 'TolFun', 1e-3, 'TolX', 1e-3);

    tic
    
    [theta2_opt] = fminsearch('gmmobjg_SOL', theta2, options);
    comp_t = toc / 60;

    theta2_full = full(sparse(theti, thetj, theta2_opt));

    varnames = {'constant'; 'price'; 'sugar'};
    mean_coef = [theta1; 0];
    sigma_coef = theta2_full(:,1);
    income_coef = theta2_full(:,2);
    BLP_results = table(varnames, mean_coef, sigma_coef, income_coef, ...
        'VariableNames', {'variable', 'mean', 'sigma', 'income'});
    
    % Define the folder where the results will be saved
    folder_path = 'results';
    
    % Save results in a unique file for every alpha_income_init
    BLP_results_name = sprintf('BLP_results_alpha_%.2f.mat', alpha_income_init);
    % Construct file's name using the value of alpha_income_init
    BLP_results_path = fullfile(folder_path, BLP_results_name);
    % Save the result in the folder specified
    save(BLP_results_path, 'BLP_results');
    
    BLP_ind_sh_name = sprintf('BLP_ind_sh_alpha_%.2f.mat', alpha_income_init);
    BLP_ind_sh_path = fullfile(folder_path, BLP_ind_sh_name);
    save(BLP_ind_sh_path, 'ind_sh');    
end
