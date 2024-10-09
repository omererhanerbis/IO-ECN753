function [f,df] = gmmobjg_SOL(theta2)
    % This function computes the GMM objective function. 
    % f is the objective, and df is the gradient.

global invA theti thetj theta1 x1 x2 IV s_jt vfull dfull cdindex cdid ns ind_sh old_theta2 delta_hat ind_sh

if max(abs(theta2-old_theta2)) < 0.01
	tol = 1e-9;
	flag = 0;
else
  	tol = 1e-6;
	flag = 1;
end

theta2_guest = full(sparse(theti,thetj,theta2));

% This loop computes the non-linear part of the utility (mu_ijt in the %%Guide)
[n, k] = size(x2);
j = size(theta2_guest,2)-1;
mu = zeros(n,ns);
for i = 1:ns
    	v_i = vfull(:,i:ns:ns);
        d_i = dfull(:,i:ns:j*ns);
 		mu(:,i) = (x2.*v_i*theta2_guest(:,1))+x2.*(d_i*theta2_guest(:,2:j+1)')*ones(k,1);
end

expmu = exp(mu);
norm = 1;
avgnorm = 1;

i = 0;

while norm > tol*10^(flag*floor(i/50)) && avgnorm > 1e-3*tol*10^(flag*floor(i/50))
    % Computes individual probabilities of choosing each brand.
    eg = expmu .* kron(ones(1, ns), delta_hat);
    sum1 = cumsum(eg);
    sum1 = sum1(cdindex, :);
    sum1(2:end, :) = diff(sum1);
    denom = 1 ./ (1 + sum1(cdid, :));
    ind_sh = eg .* denom;
    mval = delta_hat.*s_jt./(sum(ind_sh')/ns)';  
    t = abs(mval-delta_hat);
	norm = max(t);
    avgnorm = mean(t);
    delta_hat = mval;
     i = i + 1;
end
disp(['# of iterations for delta convergence:  ' num2str(i)])

if flag == 1 && max(isnan(mval)) < 1
   delta_hat = mval;
   old_theta2 = theta2;
   %save delta_hat delta_hat old_theta2
end   
delta = log(mval);

if max(isnan(delta)) == 1
	f = 1e+10;
else
    temp1 = x1'*IV;
    temp2 = delta'*IV;
    theta1 = inv(temp1*invA*temp1')*temp1*invA*temp2';
    clear temp1 temp2
    gmmresid = delta - x1*theta1;
	temp1 = gmmresid'*IV;
	f1 = temp1*invA*temp1';
    f = f1;
    clear temp1
end

disp(['GMM objective:  ' num2str(f1)])





