function BLP_results = optimize_model(alpha_income_init)

    global ind_sh

    % Verifica si se proporcionó alpha_income_init
    if nargin < 1
        error('Initial alpha_income value must be provided.');
    end

    clear global;
    clear data; 
    global invA ns x1 x2 s_jt IV vfull dfull theta1 theti thetj cdid cdindex

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

    theta2_guess = [0           0.79;
                    0.78        alpha_income_init;
                    0           0.81];

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

    save delta_hat delta_hat old_theta2
    clear delta outshr theta1_hat old_theta2 delta_hat temp sum1

    vfull = v(cdid,:);
    dfull = demogr(cdid,:);

    options = optimset('GradObj','off','MaxIter',200,'Display','iter','TolFun',0.1,'TolX',0.01);
    
    tic
    [theta2_opt] = fminsearch('gmmobjg', theta2, options);
    comp_t = toc / 60;

    theta2_full = full(sparse(theti, thetj, theta2_opt));

    varnames = {'constant'; 'price'; 'sugar'};
    mean_coef = [theta1; 0];
    sigma_coef = theta2_full(:,1);
    income_coef = theta2_full(:,2);
    BLP_results = table(varnames, mean_coef, sigma_coef, income_coef, ...
        'VariableNames', {'variable', 'mean', 'sigma', 'income'});

    % Limpia archivos temporales
    delete delta_hat.mat
    
    filename = sprintf('BLP_results_alpha_%.2f.mat', alpha_income_init);
    save(filename, 'BLP_results');
end
