function Vbar = compute_Vbar(estimates, theta_30, theta_31, theta_32, beta, x)
    % Compute value function by Bellman iteration
    k = length(x);                                 % Dimension of the state space
    U = compute_U(estimates, x);                   % Static utility
    index_theta_3 = [(1:k)' [2:k k]' [3:k k k]'];  % Mileage index
    index_A = [(1:k)' ones(k, 1)];                 % Investment index
    gamma = 0.5772156649015329;                    % Euler's gamma (approx)

    % Iterate the Bellman equation until convergence
    Vbar = zeros(k, 2);
    Vbar1 = Vbar;
    dist = 1;
    iter = 0;
    while dist > 1e-4
        V = gamma + log(sum(exp(Vbar - 0.5), 2));                 % Compute value
        expV = V(index_theta_3) * [theta_30; theta_31; theta_32]; % Compute expected value
        Vbar1 = U + beta * expV(index_A);                         % Compute v-specific
        dist = max(abs(Vbar1 - Vbar), [], 'all');                 % Check distance
        iter = iter + 1;
        Vbar = Vbar1;                                             % Update value function
    end
end