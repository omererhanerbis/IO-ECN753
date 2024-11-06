function T = compute_T(k, theta_30, theta_31, theta_32)
    % Compute transition matrix
    T = zeros(k, k, 2);

    % Conditional on not investing
    T(k, k, 1) = 1;
    for i = 1:k-2
        T(i, i, 1) = theta_30;
        T(i, i+1, 1) = theta_31;
        T(i, i+2, 1) = theta_32;
    end

    % Conditional on investing
    T(:, 1, 2) = theta_30;
    T(:, 2, 2) = theta_31;
    T(:, 3, 2) = theta_32;
    return;
end