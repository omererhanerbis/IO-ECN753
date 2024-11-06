function logL = logL_Rust(estimates, theta_30, theta_31, theta_32, beta, x, state, decision)
    % Compute log-likelihood function for Rust problem
    
    % Compute value function
    Vbar = compute_Vbar(estimates, theta_30, theta_31, theta_32, beta, x);

    % Expected choice probabilities
    EP = exp(Vbar(:, 2)) ./ (exp(Vbar(:, 1)) + exp(Vbar(:, 2)));

    % Likelihood calculation
    logL = sum(log(EP(state(decision == 1)))) + sum(log(1 - EP(state(decision == 0))));

    % Return negative log-likelihood
    logL = -logL;
end