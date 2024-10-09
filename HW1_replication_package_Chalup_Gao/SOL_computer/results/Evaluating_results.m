%% Part 2.c - Evaluating results from SOL computer

% Alpha values
alpha_values=[-33.5 -33.4 -33.3 -33.2 -33.1 -33.0 -32.9 -32.8 -32.7 -32.6];

% List of .mat files (assuming they are in the current directory)
mat_files = dir('BLP_results_alpha*.mat');

% Number of results (the number of .mat files)
num_results = length(mat_files);
price_mean_coefs = zeros(num_results, 1);

% Loop over each .mat file and extract the relevant information
for i = 1:num_results
    % Load the .mat file
    mat_file = load(mat_files(i).name);
    
    % Assuming the loaded .mat file contains a structure `BLP_results`
    % and that the coefficient for 'price' is stored in mean(2,1)
    BLP_results = mat_file.BLP_results;
    price_mean_coefs(i) = BLP_results.mean(2, 1);
end

figure;
plot(alpha_values, price_mean_coefs', '-o');
xlabel('Initial alpha\_income');
ylabel('Mean-price coefficient');
ylim([-3.2, -2.9]);
title('Optimization results');
saveas(gcf, 'Optimization_results.png');