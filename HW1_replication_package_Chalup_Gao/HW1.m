% Course: PhD Industrial Organization
% Proffesor: Nicholas Vreugdenhil
% Student: Chalup, Miguel & Gao, Titus
% Data: product_data.csv contains data about market shares, prices, and product characteristics.
% Homework 1
% Data: 09/10/2024

%% Part 1.a - OLS Regression

clear;
clear variables;
clear global;

% Load the data from CSV file
data = readtable('product_data.csv');

% Create categorical variables for product and market IDs
data.products = categorical(data.product_ids);
data.markets = categorical(data.market_ids);

% Display summary statistics
%summary(data);

% Compute the market share sum by market and generate the share0 variable
marketGroup = findgroups(data.markets);
sum_shares = splitapply(@sum, data.shares, marketGroup);
data.share0 = 1 - sum_shares(marketGroup);

% Generate the log ratio of shares
data.lrat_sj_s0 = log(data.shares ./ data.share0);

% Compute summary statistics by product
grpStats = varfun(@mean, data, 'InputVariables', {'shares', 'prices', 'sugar'}, ...
                  'GroupingVariables', 'products');

% Display statistics
disp(grpStats);

% OLS regression
X = [data.prices data.sugar];
y = data.lrat_sj_s0;
ols_logit_model = fitlm(X, y);

% Display OLS coefficients
disp(ols_logit_model.Coefficients);
beta_p_logit = ols_logit_model.Coefficients.Estimate(2);

% Save coefficients
writetable(ols_logit_model.Coefficients, 'ols_logit_results.csv');

%% Part 1.b - 2SLS Regression

% Instrumental Variables (2SLS)
Z = data{:, {'demand_instruments0', 'demand_instruments1', 'demand_instruments2', 'demand_instruments3', ...
             'demand_instruments4', 'demand_instruments5', 'demand_instruments6', 'demand_instruments7', ...
             'demand_instruments8', 'demand_instruments9', 'demand_instruments10', 'demand_instruments11', ...
             'demand_instruments12', 'demand_instruments13', 'demand_instruments14', 'demand_instruments15', ...
             'demand_instruments16', 'demand_instruments17', 'demand_instruments18', 'demand_instruments19'}};

% First stage regression: Regress price on instruments
first_stage = fitlm(Z, data.prices);

% Predicted prices from the first stage
predicted_prices = predict(first_stage, Z);

% Second stage regression: Regress log ratios on predicted prices and sugar
X_iv = [predicted_prices data.sugar];
iv_model = fitlm(X_iv, y);

% Display 2SLS coefficients
disp(iv_model.Coefficients);
beta_p_ivlogit = iv_model.Coefficients.Estimate(2);

% Save coefficients to a LaTeX file (this would need a LaTeX export function)
writetable(iv_model.Coefficients, 'IV_logit_results.csv');

%% Part 1.c - Own-price Elasticities and Scatterplot

% Compute own-price elasticities for each product
data.elas_ivlogit = beta_p_ivlogit * (1 - data.shares) .* data.prices;


% Display summary statistics for elasticities by product
elasticity_stats_logit = varfun(@mean, data, 'InputVariables', 'elas_ivlogit', 'GroupingVariables', 'products');
%disp(elasticity_stats_logit);

% Check if market_ids is a cell array and convert to string array for comparison
if iscell(data.market_ids)
    data.market_ids = string(data.market_ids);
end

% Filter data for the specific market 'C01Q1'
market_filter = strcmp(data.market_ids, 'C01Q1');

% Calculate mean and standard deviation of elasticities for each product in market 'C01Q1'
elas_stats_logit = varfun(@mean, data(market_filter, :), 'InputVariables', 'elas_ivlogit', 'GroupingVariables', 'product_ids');
elas_stats_std_logit = varfun(@std, data(market_filter, :), 'InputVariables', 'elas_ivlogit', 'GroupingVariables', 'product_ids');
%disp('Elasticity Statistics for Market C01Q1:');
%disp(elas_stats_logit);
%disp(elas_stats_std_logit);

% Scatterplot of elasticities vs. prices
scatter(data.prices(market_filter), data.elas_ivlogit(market_filter));
xlabel('Price');
ylabel('Elasticity');
saveas(gcf, 'scatter.png');

%% Part 2.a - BLP Model

% This code follow appendix to Nevo (2000) - "A Practitioner’s Guide to
% Estimation of Random-Coefficients Logit Models of Demand"

clear global;
clear data; 
global invA ns x1 x2 s_jt IV vfull dfull theta1 theti thetj cdid cdindex ind_sh

% Load the data from CSV file
product_data = readtable('product_data.csv');
agent_data = readtable('agent_data.csv');

ns = 20;       % number of simulated "indviduals" per market
nmkt = 94;     % number of markets
nbrn = 24;     % number of products per market
n_inst = 20;    %Number of instruments for price
%ns=20, nmkt=94, nbrn=24, thus we have 24*94=2256 observations

% The vector below assigns each observation to the corresponding market.
cdid = kron((1:nmkt)',ones(nbrn,1));
% The vector below contains the index of the last observation for each market.
cdindex = (nbrn:nbrn:nbrn*nmkt)';

% Instruments
IV = (product_data(:, 6:26)); % Includes sugar
IV = IV{:,:}; % Extract all data from the table

% Individuals data
v = reshape(agent_data.nodes0, 20, []).'; % Reshape the 'income' column to 20 rows and as many columns as there are markets, then transpose
demogr = reshape(agent_data.income, 20, []).'; % Reshape the 'income' column to 20 rows and as many columns as there are markets, then transpose

s_jt = product_data.shares(:, :);
x1 =[ones(nmkt*nbrn, 1) product_data.prices(:, :)];
x2= [ones(nmkt*nbrn, 1)  product_data.prices(:, :) product_data.sugar(:, :)];

%Guess of theta2
    theta2_guess = [0                      4.5;
                    1                      -33;
                    0                    0.15];
            
% Create a vector of the non-zero elements in the above matrix, along with 
% their corresponding row and column indices. This simplifies passing values 
% to the functions below.
[theti, thetj, theta2]=find(theta2_guess);
theti = double(theti);
thetj = double(thetj);
theta2 = double(theta2);

%The model have 9 parameters, 3 linear and 6 non linear
%There are 6 parameters to be estimated
%3 parameters are set to 0 by assumption                                                           ')

% compute the outside good market share by market:
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

%the next command creates a new file, delta_hat.mat, with delta_hat and old_theta2 in it, and then clears out the old old_theta2 from memory. 
save delta_hat delta_hat old_theta2
clear delta outshr theta1_hat old_theta2 delta_hat temp sum1

vfull = v(cdid,:);
dfull = demogr(cdid,:);

%Below, I set the maximum number of iterations for the main optimization command. 

options = optimset('GradObj', 'off', 'MaxIter', 1e20, 'MaxFunEvals', 1e20, 'Display', 'iter', 'TolFun', 1e-3, 'TolX', 1e-3);

tic

[theta2,fval,exitflag,output]  = fminsearch('gmmobjg',theta2, options);

comp_t = toc/60;

theta2 = full(sparse(theti,thetj,theta2));

%Display results
varnames = {'constant'; 'price'; 'sugar'};
mean_coef = [theta1; 0];
sigma_coef = theta2(:,1);
income_coef = theta2(:,2);
BLP_results = table(varnames, mean_coef, sigma_coef, income_coef, ...
    'VariableNames', {'variable', 'mean', 'sigma', 'income'});

writetable(BLP_results, 'BLP_results.csv');

delete delta_hat.mat

%% Part 2.b - Own-price Elasticities and Scatterplot

% Check if market_ids is a cell array and convert to string array for comparison
if iscell(product_data.market_ids)
    product_data.market_ids = string(product_data.market_ids);
end

% Filter data for the specific market 'C01Q1'
market_filter = strcmp(product_data.market_ids, 'C01Q1');
product_data_C01Q1 = product_data(market_filter, :);

agent_filter = strcmp(agent_data.market_ids, 'C01Q1');
agent_data_C01Q1 = agent_data(agent_filter, :);

ind_sh_C01Q1 = ind_sh(market_filter, :);

% Compute own-price elasticities for C01Q1

product_data_C01Q1.elas_BLP = (product_data_C01Q1.prices./product_data_C01Q1.shares).*(((theta1(2) + theta2(2,1)*agent_data_C01Q1.nodes0 + theta2(2,2)*agent_data_C01Q1.income)'*(ind_sh_C01Q1-ind_sh_C01Q1.^2)')/20)';

% Display summary statistics for elasticities by product
elasticity_stats_BLP = varfun(@mean, product_data_C01Q1, 'InputVariables', 'elas_BLP', 'GroupingVariables', 'product_ids');
%disp(elasticity_stats_BLP);

% Scatterplot of elasticities vs. prices
figure();
scatter(product_data_C01Q1.prices, product_data_C01Q1.elas_BLP);
xlabel('Price');
ylabel('Elasticity');
saveas(gcf, 'scatter_BLP.png');

