function value = NLP(means,variances,yTrue)
% value = NLP(means,variances,yTrue)
%
% Negative log predictive denisty
%
value = 0.5 * mean( log(2*pi*variances) + ( (means - yTrue).^2 ) ./ variances );