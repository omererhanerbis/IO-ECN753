function logL = logL_HM(estimates, beta, x, state, decision, T, CCP)
    % Compute log-likelihood function for HM problem

    % Compute static utility
    U = compute_U(estimates, x);

    % Expected value by inversion
    EV_ = HM_inversion(CCP, T, U, beta);

    % Compute implied choice probabilities
    EP_ = from_EV_to_EP(EV_, T, U, beta);

    % Compute the log-likelihood
    logL = sum(log(EP_(state(decision == 1)))) + sum(log(1 - EP_(state(decision == 0)) + 0.0001));

    % Return negative log-likelihood
    logL = -logL;
end
