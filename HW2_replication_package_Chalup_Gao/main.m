clear

%% Q1 (Rust, 1987)

% Set parameters
estimates = [2.293, 10.075];
estimates0 = [2, 10];
theta_30 = 0.3919;
theta_31 = 0.5953;
theta_32 = 1 - 0.3919 - 0.5953;
beta = 0.9999;

% State space
k = 90;
x = (1:k)';

% Read data
df = readtable('group_4.csv');
df.state = df.state + 1;

% Display the first 6 rows with 4 decimal digits
disp(df(1:6, :));

% Optimize using fminunc (MATLAB's unconstrained optimization function)
options = optimoptions('fminunc', 'Display', 'iter');
estimates_R = fminunc(@(y) logL_Rust(y, theta_30, theta_31, theta_32, beta, x, df.state, df.decision), estimates0, options);

% Display the estimated thetas
fprintf('Estimated theta_11 and RC: [%.4f, %.4f] (true = [%.4f, %.4f])\n', ...
    estimates_R(1), estimates_R(2), estimates(1), estimates(2));

%% Q2 (Hotz & Miller, 1993)

% Adjust dimensions
k2 = 78;
x2 = (1:k2)';

% Estimate CCP
P = arrayfun(@(i) mean(df.decision(df.state == i)), x2);
P = P + 0.0001;

% Combine into matrix (1 - P, P)
CCP = [(1 - P) P];

% Compute T
T = compute_T(k2, theta_30, theta_31, theta_32);

% Conditional on not investing
T(:,:,1)

% T Conditional on investing
T(:,:,2)

% Optimize using fminunc
estimates_HM = fminunc(@(y) logL_HM(y, beta, x2, df.state, df.decision, T, CCP), estimates0, options);

% Display the estimated thetas
fprintf('Estimated theta_11 and RC: [%.4f, %.4f] (true = [%.4f, %.4f])\n', ...
    estimates_HM(1), estimates_HM(2), estimates(1), estimates(2));

%% Q2 (Hotz & Miller, 1993) with  binned state of mileage in 50000 mile increments

state3 = 0:50000:max(df.mileage) + 50000;
df.state3 = discretize(df.mileage, state3);

% Adjust dimensions
k3 = 8;
x3 = (1:k3)';

% Estimate CCP
P = arrayfun(@(i) mean(df.decision(df.state3 == i)), x3);
P = P + 0.0001;

% Combine into matrix (1 - P, P)
CCP = [(1 - P) P];

% Compute T
T = compute_T(k3, theta_30, theta_31, theta_32);

% Conditional on not investing
T(:,:,1)

% T Conditional on investing
T(:,:,2)

% Optimize using fminunc
estimates_HM2 = fminunc(@(y) logL_HM(y, beta, x3, df.state3, df.decision, T, CCP), estimates0, options);

% Display the estimated thetas
fprintf('Estimated theta_11 and RC: [%.4f, %.4f] (true = [%.4f, %.4f])\n', ...
    estimates_HM2(1), estimates_HM2(2), estimates(1), estimates(2));

%% Q2 (Hotz & Miller, 1993) assumming CCP has a linear behaviour

% Estimate CCP
logit = fitglm(df, 'decision ~ state', 'Distribution', 'binomial', 'Link', 'logit');
disp(logit);

states = table(x2, 'VariableNames', {'state'});

% Usar el modelo estimado para predecir las probabilidades
P = predict(logit, states);

% Combine into matrix (1 - P, P)
CCP = [(1 - P) P];

% Compute T
T = compute_T(k2, theta_30, theta_31, theta_32);

% Conditional on not investing
T(:,:,1)

% T Conditional on investing
T(:,:,2)

% Optimize using fminunc
estimates_HM3 = fminunc(@(y) logL_HM(y, beta, x2, df.state, df.decision, T, CCP), estimates0, options);

% Display the estimated thetas
fprintf('Estimated theta_11 and RC: [%.4f, %.4f] (true = [%.4f, %.4f])\n', ...
    estimates_HM3(1), estimates_HM3(2), estimates(1), estimates(2));