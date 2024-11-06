function EV_ = HM_inversion(CCP, T, U, beta)
    % Perform HM inversion

    % Euler's gamma constant
    gamma = 0.5772156649015329; % Approximation of Euler's gamma

    % Compute LHS (to be inverted)
    LEFT = eye(size(T, 1)) - beta * (CCP(:, 1) .* T(:, :, 1) + CCP(:, 2) .* T(:, :, 2));

    % Compute RHS (not to be inverted)
    RIGHT = gamma + sum(CCP .* (U - log(CCP)), 2);

    % Compute EV by matrix inversion
    EV_ = LEFT \ RIGHT; % Equivalent to inv(LEFT) * RIGHT in MATLAB

    % Return as a vector
    EV_ = EV_(:);
end
