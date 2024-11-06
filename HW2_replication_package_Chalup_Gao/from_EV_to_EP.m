function EP_ = from_EV_to_EP(EV_, T, U, beta)
    % Compute expected policy from expected value

    % Compute the exponentiated utilities
    E = exp(U + beta * [(T(:,:,1) * EV_), (T(:,:,2) * EV_)]);

    % Compute the expected policy
    EP_ = E(:, 2) ./ sum(E, 2);

    % Return as a column vector
    EP_ = EP_(:);
end
