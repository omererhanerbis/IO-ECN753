function U = compute_U(estimates, x)
    % Compute static utility
    u1 = -0.001*estimates(1)*x;            % Utility of not investing
    u2 = -estimates(2)*ones(size(x));            % Utility of investing
    U = [u1 u2];                       % Combine in a matrix
end
