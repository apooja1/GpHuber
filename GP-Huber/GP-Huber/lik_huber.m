function lik = lik_huber(varargin)
%LIK_GAUSSIAN  Create a Gaussian likelihood structure
%
%  Description
%    LIK = LIK_GAUSSIAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a Gaussian likelihood structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    LIK = LIK_GAUSSIAN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a likelihood function structure with the named
%    parameters altered with the specified values.
%
%    Parameters for Gaussian likelihood function [default]
%      sigma2       - variance [0.1]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%      n            - number of observations per input (See using average
%                     observations below)
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    Using average observations
%    The lik_gaussian can be used to model data where each input vector is
%    attached to an average of varying number of observations. That is, we
%    have input vectors x_i, average observations y_i and sample sizes n_i.
%    Each observation is distributed  
%
%        y_i ~ N(f(x_i), sigma2/n_i)
%
%    The model is constructed as lik_gaussian('n', n), where n is the same
%    length as y and collects the sample sizes. 
%
%  See also
%    GP_SET, PRIOR_*, LIK_*

% Internal note: Because Gaussian noise can be combined
% analytically to the covariance matrix, lik_gaussian is internally
% little between lik_* and gpcf_* functions.
%
% Copyright (c) 2007-2017 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_HUBER';
  ip.addOptional('lik', [], @(x) isstruct(x) || isempty(x));
  ip.addParamValue('sigma2',0.1, @(x) isscalar(x) && x>=0);
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
%   ip.addParamValue('n',[], @(x) isreal(x) && all(x>0));
  ip.addParamValue('weights',[], @(x) isreal(x) && all(x>0));
%   ip.addParamValue('b',[], @(x) isreal(x) && all(x>0));
%   ip.addParamValue('epsilon',[], @(x) isreal(x) && all(x>0));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Huber';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Huber')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
%   if init || ~ismember('n',ip.UsingDefaults)
%     lik.n = ip.Results.n;
%   end
  if init ||  ~ismember('weights',ip.UsingDefaults)
     lik.weights = ip.Results.weights; 
  end
% 
%   if init ||  ~ismember('b',ip.UsingDefaults)
%      lik.b = ip.Results.b; 
%   end
% 
%   if init ||  ~ismember('epsilon',ip.UsingDefaults)
%      lik.epsilon = ip.Results.epsilon; 
%   end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_huber_pak;
    lik.fh.unpak = @lik_huber_unpak;
    lik.fh.ll = @lik_huber_ll;
    lik.fh.llg = @lik_huber_llg;    
    lik.fh.llg2 = @lik_huber_llg2;
    lik.fh.llg3 = @lik_huber_llg3;
    lik.fh.lp = @lik_huber_lp;
    lik.fh.lpg = @lik_huber_lpg;
%     lik.fh.cfg = @lik_huber_cfg;
    lik.fh.tiltedMoments = @lik_huber_tiltedMoments;
%     lik.fh.trcov  = @lik_huber_trcov;
%     lik.fh.trvar  = @lik_huber_trvar;
    lik.fh.predy = @lik_huber_predy;    
    lik.fh.siteDeriv = @lik_huber_siteDeriv;
     lik.fh.invlink = @lik_huber_invlink;
%       lik.fh.optimizef=@lik_huber_optimizef;
    lik.fh.recappend = @lik_huber_recappend;
  end

end

function [w, s, h] = lik_huber_pak(lik)
%LIK_GAUSSIAN_PAK  Combine likelihood parameters into one vector.
%
%  Description
%    W = LIK_GAUSSIAN_PAK(LIK) takes a likelihood structure LIK
%    and combines the parameters into a single row vector W.
%    This is a mandatory subfunction used for example in energy 
%    and gradient computations.
%
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]'
%     
%  See also
%    LIK_GAUSSIAN_UNPAK

  w = []; s = {}; h=[];
  if ~isempty(lik.p.sigma2)
    w = [w log(lik.sigma2)];
    s = [s; 'log(gaussian.sigma2)'];
    h = [h 0];
    % Hyperparameters of sigma2
    [wh, sh, hh] = lik.p.sigma2.fh.pak(lik.p.sigma2);
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end    

end

function [lik, w] = lik_huber_unpak(lik, w)
%LIK_GAUSSIAN_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_GAUSSIAN_UNPAK(W, LIK) takes a likelihood structure
%    LIK and extracts the parameters from the vector W to the LIK
%    structure. This is a mandatory subfunction used for example 
%    in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]'
%
%  See also
%    LIK_GAUSSIAN_PAK
  
  if ~isempty(lik.p.sigma2)
    lik.sigma2 = exp(w(1));
    w = w(2:end);
    % Hyperparameters of sigma2
    [p, w] = lik.p.sigma2.fh.unpak(lik.p.sigma2, w);
    lik.p.sigma2 = p;
  end
end


function logLik = lik_huber_ll(lik, y, f, ~) 
% epsilon=lik.epsilon;
% b=lik.b;

%LIK_GAUSSIAN_LL    Log likelihood
%
%  Description
%    E = LIK_GAUSSIAN_LL(LIK, Y, F, Z) takes a likelihood data
%    structure LIK, incedence counts Y, expected counts Z, and
%    latent values F. Returns the log likelihood, log p(y|f,z).
%    This subfunction is needed when using Laplace approximation
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC)
%    computations.
%
%  See also
%    LIK_GAUSSIAN_LLG, LIK_GAUSSIAN_LLG3, LIK_GAUSSIAN_LLG2, GPLA_E
  epsilon=0.45;
  b=0.5;
  s2 = lik.sigma2;
  sigma=sqrt(s2);
  r = (f-y);
%   weights=ones(length(r),1); 
 weights=lik.weights;
 rs=r./(weights.*sigma);
rho=@(rs) b.^2*(sqrt(1+(rs./b).^(2))-1);
%   logLik =  sum(-0.5 * r2./s2 - 0.5*log(s2) - 0.5*log(2*pi));
logLik1= -rho(rs) -log(sigma) - 0.5.*log(2.*pi)+log(1-epsilon);
logLik=sum(logLik1);
end

function llg = lik_huber_llg(lik, y, f, param, ~)
%LIK_GAUSSIAN_LLG    Gradient of the log likelihood
%
%  Description 
%    G = LIK_GAUSSIAN_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z
%    and latent values F. Returns the gradient of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    'param' or 'latent'. This subfunction is needed when using 
%    Laplace approximation or MCMC for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    LIK_GAUSSIAN_LL, LIK_GAUSSIAN_LLG2, LIK_GAUSSIAN_LLG3, GPLA_E
  
% epsilon=lik.epsilon;
% weights=lik.weights;
% b=lik.b;
 
switch param
    case 'param'
        % there is also correction due to the log transformation
          s2 = lik.sigma2;
          sigma=sqrt(s2);
          epsilon=0.45;
          b=0.5;
          r = (y-f);
%           c1=1+5/(length(f(:,1))-length(f(1,:)));
%           s1 = 1.4826*(c1)*median(abs(r));
%           rs=r./(weights.*sigma);
%             weights=ones(length(r),1); 
            weights=lik.weights;
%           llg=sum((-1./sigma)+b.^(2).*r.^(2)./((weights.^(2).*sigma.^(3)).*(sqrt((r.^2./weights.^(2).*s2))+1)));
% llg=sum((b.^3.*f.*(y - (b.*f.*sigma)./weights))./(weights.*((y - (b.*f.*sigma)./weights).^2 + 1).^(1/2))) - 1/sigma;

 llg=sum((f - y).^2./(sigma^3.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2))) - 1/sigma;


    case 'latent'
         epsilon=0.45;
         b=0.5;
        s2 = lik.sigma2;
        sigma=sqrt(s2);
         r = (y-f);
         c1=1+5/(length(f(:,1))-length(f(1,:)));
         s1 = 1.4826*(c1)*median(abs(r));
%          rs=r./(weights.*sigma);
% weights=ones(length(r),1);
weights=lik.weights;
% llg=(r.*b.^2)./(weights.*sigma.*sqrt(sigma.^2.*weights.^2+r.^2));
 llg=-(f - y)./(sigma^2.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2));
 
% llg=sum(llg1);       
end
end


function llg2 = lik_huber_llg2(lik, y, f, param, ~)
%LIK_GAUSSIAN_LLG2  Second gradients of the log likelihood
%
%  Description        
%    G2 = LIK_GAUSSIAN_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z,
%    and latent values F. Returns the Hessian of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. G2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction
%    is needed when using Laplace approximation or EP for inference 
%    with non-Gaussian likelihoods.

%
%  See also
%    LIK_GAUSSIAN_LL, LIK_GAUSSIAN_LLG, LIK_GAUSSIAN_LLG3, GPLA_E
% epsilon=lik.epsilon;
% weights=lik.weights;
% b=lik.b;
   
switch param
    case 'latent'
      s2 = lik.sigma2;
      sigma=sqrt(s2);
      epsilon=0.45;
      b=0.5;
      r = (y-f);
%       c1=1+5/(length(f(:,1))-length(f(1,:)));
%       s1 = 1.4826*(c1)*median(abs(r));
%       rs=r./(weights.*sigma);
%       weights=ones(length(r),1);
       weights=lik.weights;
%       llg2=-b.^2.*s2.*weights.^(2)./(s2.*weights.^(2)+f-2.*f.*y+y.^(2)).^(3/2);
     llg2=-1./(sigma.^2.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(3/2));

    case 'latent+param'
        % there is also correction due to the log transformation
          s2 = lik.sigma2;
          sigma=sqrt(s2);
          epsilon=0.45;
          b=0.5;
          r = (y-f);
%           c1=1+5/(length(f(:,1))-length(f(1,:)));
%           s1 = 1.4826*(c1)*median(abs(r));

% weights=ones(length(r),1);
 weights=lik.weights;
% llg2=(1/s2)-b^(2).*abs(sigma).*(3*weights.^2.*r.^2.*s2+2*r.^4)./(sigma.^4.*abs(weights).*(weights.^2.*s2+r.^2).^(3/2));
% llg2=(weights.^2.*((y - (b.*f.*sigma)./weights).^2 + 1).^(3/2) - b^4.*f.^2*sigma.^2)./(sigma.^2.*weights.^2.*((y - (b.*f.*sigma)./weights).^2 + 1).^(3/2));
% llg2=1./sigma^2 - (3.*(f - y).^2)./(sigma^4*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2)) + (f - y).^4./(b^2.*sigma.^6.*weights.^4.*((f - y).^2/(b^2*sigma.^2*weights.^2) + 1).^(3/2));
llg2=((f - y).*(2*b^2*sigma^2.*weights.^2 + f.^2 - 2.*f.*y + y.^2))./(b^2*sigma^5.*weights.^4.*((f - y).^2./(b^2.*sigma^2.*weights.^2) + 1).^(3/2));

end
end    

function llg3 = lik_huber_llg3(lik, y, f, param, z)


%LIK_GAUSSIAN_LLG3  Third gradients of the log likelihood
%
%  Description
%    G3 = LIK_GAUSSIAN_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, incedence counts Y, expected counts Z
%    and latent values F and returns the third gradients of the
%    log likelihood with respect to PARAM. At the moment PARAM
%    can be only 'latent'. G3 is a vector with third gradients.
%    This subfunction is needed when using Laplace approximation 
%    for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_GAUSSIAN_LL, LIK_GAUSSIAN_LLG, LIK_GAUSSIAN_LLG2, GPLA_E, GPLA_G

switch param
    case 'latent'
        s2 = lik.sigma2;
        sigma=sqrt(s2); 
        epsilon=0.45;
          b=0.5;
%         llg3 = zeros(size(y));

% d/df(-(b^2 s^2 w^2)/(s^2 w^2 + f - 2 f y + y^2)^1.5) = (1.5 b^2 s^2 w^2 (1 - 2 y))/(-2 f y + f + s^2 w^2 + y^2)^2.5
%      weights=ones(length(y),1);
      weights=lik.weights;
%      llg3= (1.5*b.^2 .*sigma.^2.* weights.^2 .*(1 - 2.* y))./(-2.* f.* y + f + sigma.^2 .*weights.^2 + y.^2).^2.5;
 llg3=(3.*f - 3.*y)./(b.^2.*sigma.^4.*weights.^4.*((b.^2.*sigma.^2.*weights.^2 + f.^2 - 2*f.*y + y.^2)./(b^2*sigma^2.*weights.^2)).^(5/2));

    case 'latent2+param'
        s2 = lik.sigma2;
        sigma=sqrt(s2);
         epsilon=0.45;
         b=0.5;
         r=y-f;
        % there is also correction due to the log transformation
%         llg3 = ones(size(y)) ; 
%         weights=ones(length(r),1);
         weights=lik.weights;
%         llg3= -2/sigma^3-b^2.*r.^2.*((2.*r.^4+5.*r.^2.*s2.*weights.^2).*(sigma.*abs(sigma)-4*abs(sigma))+3*sigma.^4.*weights.^4.*(sigma.*abs(sigma)-5*abs(sigma)))./(sigma.^5.*abs(weights).*(r.^2+s2.*weights.^2).^(5/2));
% llg3= (12.*(f - y).^2)./(sigma^5.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2)) - 2./sigma^3 - (9.*(f - y).^4)./(b^2*sigma^7.*weights.^4.*((f - y).^2/(b^2.*sigma^2.*weights.^2) + 1).^(3/2)) + (3.*(f - y).^6)./(b^4*sigma^9.*weights.^6.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(5/2));
llg3=(2*b^2*sigma^2.*weights.^2 - f.^2 + 2.*f.*y - y.^2)./(b.^2.*sigma^5.*weights.^4.*((f - y).^2./(b^2.*sigma^2.*weights.^2) + 1).^(5/2));

end
end


function lp = lik_huber_lp(lik)
%LIK_GAUSSIAN_LP  Evaluate the log prior of likelihood parameters
%
%  Description
%    LP = LIK_T_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters.
%    This subfunctions is needed when there are likelihood
%    parameters.
%
%  See also
%    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_G, GP_E

  lp = 0;

  if ~isempty(lik.p.sigma2)
    likp=lik.p;
    lp = likp.sigma2.fh.lp(lik.sigma2, likp.sigma2) + log(lik.sigma2);
  end
end

function lpg = lik_huber_lpg(lik)
%LIK_GAUSSIAN_LPG  Evaluate gradient of the log prior with respect
%                  to the parameters.
%
%  Description
%    LPG = LIK_GAUSSIAN_LPG(LIK) takes a Gaussian likelihood
%    function structure LIK and returns LPG = d log (p(th))/dth,
%    where th is the vector of parameters. This subfunction is 
%    needed when there are likelihood parameters.
%
%  See also
%    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G

  lpg = [];

  if ~isempty(lik.p.sigma2)
    likp=lik.p;
    
    lpgs = likp.sigma2.fh.lpg(lik.sigma2, likp.sigma2);
    lpg = lpgs(1).*lik.sigma2 + 1;
    if length(lpgs) > 1
      lpg = [lpg lpgs(2:end)];
    end            
  end
end



% function DKff = lik_gaussian_cfg(lik, x, x2)
% %LIK_GAUSSIAN_CFG  Evaluate gradient of covariance with respect to
% %                 Gaussian noise
% %
% %  Description
% %    Gaussian likelihood is a special case since it can be
% %    analytically combined with covariance functions and thus we
% %    compute gradient of covariance instead of gradient of likelihood.
% %
% %    DKff = LIK_GAUSSIAN_CFG(LIK, X) takes a Gaussian likelihood
% %    function structure LIK, a matrix X of input vectors and
% %    returns DKff, the gradients of Gaussian noise covariance
% %    matrix Kff = k(X,X) with respect to th (cell array with
% %    matrix elements). This subfunction is needed only in Gaussian 
% %    likelihood.
% %
% %    DKff = LIK_GAUSSIAN_CFG(LIK, X, X2) takes a Gaussian
% %    likelihood function structure LIK, a matrix X of input
% %    vectors and returns DKff, the gradients of Gaussian noise
% %    covariance matrix Kff = k(X,X) with respect to th (cell
% %    array with matrix elements). This subfunction is needed 
% %    only in Gaussian likelihood.
% %
% %  See also
% %    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G
% 
%   DKff = {};
%   if ~isempty(lik.p.sigma2)
%       if isempty(lik.n)
%           DKff{1}=lik.sigma2;
%       else
%           n=size(x,1);
%           DKff{1} = sparse(1:n, 1:n, lik.sigma2./lik.n, n, n);
%       end
%   end
% end

% function DKff  = lik_gaussian_ginput(lik, x, t, g_ind, gdata_ind, gprior_ind, varargin)
% %LIK_GAUSSIAN_GINPUT  Evaluate gradient of likelihood function with 
% %                     respect to x.
% %
% %  Description
% %    DKff = LIK_GAUSSIAN_GINPUT(LIK, X) takes a likelihood
% %    function structure LIK, a matrix X of input vectors and
% %    returns DKff, the gradients of likelihood matrix Kff =
% %    k(X,X) with respect to X (cell array with matrix elements).
% %    This subfunction is needed only in Gaussian likelihood.
% %
% %    DKff = LIK_GAUSSIAN_GINPUT(LIK, X, X2) takes a likelihood
% %    function structure LIK, a matrix X of input vectors and
% %    returns DKff, the gradients of likelihood matrix Kff =
% %    k(X,X2) with respect to X (cell array with matrix elements).
% %    This subfunction is needed only in Gaussian likelihood.
% %
% %  See also
% %    LIK_GAUSSIAN_PAK, LIK_GAUSSIAN_UNPAK, LIK_GAUSSIAN_E, GP_G
% 
% end

% function C = lik_gaussian_trcov(lik, x)
% %LIK_GAUSSIAN_TRCOV  Evaluate training covariance matrix
% %                    corresponding to Gaussian noise
% %
% %  Description
% %    C = LIK_GAUSSIAN_TRCOV(GP, TX) takes in covariance function
% %    of a Gaussian process GP and matrix TX that contains
% %    training input vectors. Returns covariance matrix C. Every
% %    element ij of C contains covariance between inputs i and j
% %    in TX. This subfunction is needed only in Gaussian likelihood.
% %
% %  See also
% %    LIK_GAUSSIAN_COV, LIK_GAUSSIAN_TRVAR, GP_COV, GP_TRCOV
% 
%   [n, m] =size(x);
%   n1=n+1;
% 
%   if isempty(lik.n)
%       C = sparse(1:n,1:n,ones(n,1).*lik.sigma2,n,n);
%   else  
%       C = sparse(1:n, 1:n, lik.sigma2./lik.n, n, n);
%   end
% 
% end

% function C = lik_gaussian_trvar(lik, x)
% %LIK_GAUSSIAN_TRVAR  Evaluate training variance vector
% %                    corresponding to Gaussian noise
% %
% %  Description
% %    C = LIK_GAUSSIAN_TRVAR(LIK, TX) takes in covariance function
% %    of a Gaussian process LIK and matrix TX that contains
% %    training inputs. Returns variance vector C. Every element i
% %    of C contains variance of input i in TX. This subfunction is 
% %    needed only in Gaussian likelihood.
% %
% %
% %  See also
% %    LIK_GAUSSIAN_COV, GP_COV, GP_TRCOV
% 
%   [n, m] =size(x);
%   if isempty(lik.n)
%       C=repmat(lik.sigma2,n,1);
%   else
%       C=lik.sigma2./lik.n(:);
%   end
% 
% end


function [lpy, Ey, Vary] = lik_huber_predy(lik, Ef, Varf, yt, zt)   %requires attention
%LIK_Gaussian_PREDY    Returns the predictive mean, variance and density of y
%
%  Description  
%    LPY = LIK_POISSON_PREDY(LIK, EF, VARF YT, ZT)
%    Returns also the predictive density of YT, that is 
%        p(yt | y,zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the incedence counts YT, expected counts ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_POISSON_PREDY(LIK, EF, VARF, YT, ZT) 
%    takes a likelihood structure LIK, posterior mean EF and 
%    posterior variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also 
%    GPLA_PRED, GPEP_PRED, GPMC_PRED
  sigma2 = lik.sigma2;
  sigma = sqrt(sigma2);
%   epsilon=lik.epsilon;
%   weights=lik.weights;
%   b=lik.b;
     epsilon=0.45;
     b=0.5;
  EVary = zeros(size(Ef));
  VarEy = zeros(size(Ef)); 
   Ey = Ef;
   Vary = EVary + VarEy;
  lpy = zeros(length(yt),1);

  if (size(Ef,2) > 1) && size(yt,2) == 1
    % Approximate integral with sum of grid points when using corrected
    % marginal posterior
    for i2=1:length(yt)
      py = h_pdf(yt(i2), Ef(i2,:), sigma);
      pf = Varf(i2,:)./sum(Varf(i2,:));
      lpy(i2) = log(sum(py.*pf));
    end
  else
    for i2 = 1:length(yt)

%       mean_app = Ef(i2);
%       sigm_app = sqrt(Varf(i2));
%       pd = @(f) h_pdf(y(i2),f, sigma, weights,b, s1).*norm_pdf(f,Ef(i2),sqrt(Varf(i2)));
%       lpy(i2) = log(quadgk(pd, mean_app - 12*sigm_app, mean_app + 12*sigm_app));

      [pdf, minf,maxf]=init_huber_norm(yt(i2),Ef(i2),Varf(i2),sigma);
      lpy(i2) = log(quadgk(pdf, minf, maxf));
    end
  end
  
end

% 
% function [logM_0, m_1, sigm2hati1] = lik_huber_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)  %perhaps for classification
% %LIK_PROBIT_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
% %
% %  Description
% %    [M_0, M_1, M2] = LIK_PROBIT_TILTEDMOMENTS(LIK, Y, I, S2,
% %    MYY) takes a likelihood structure LIK, class labels Y, index
% %    I and cavity variance S2 and mean MYY. Returns the zeroth
% %    moment M_0, mean M_1 and variance M_2 of the posterior
% %    marginal (see Rasmussen and Williams (2006): Gaussian
% %    processes for Machine Learning, page 55). This subfunction 
% %    is needed when using EP for inference with non-Gaussian 
% %    likelihoods.
% %
% %  See also
% %    GPEP_E
% 
% %   m_1=myy_i;
% %   sigm2hati1=sigm2_i;
% %   logM_0=zeros(size(y));
%   
% s2 = lik.sigma2;
% tau = 1./s2 + 1./sigm2_i;
% w = (y(i1)./s2 + myy_i./sigm2_i)./tau;
% %Zi = 1 ./( sqrt( 2.*pi.*(s2+sigm2_i) ) ) .* exp( -0.5*(y(i1)-myy_i).^2./(s2+sigm2_i) );
% 
% m_1 = w;
% sigm2hati1 = 1./tau;
% %logM_0 = log(Zi)
% logM_0 = -0.5*log( 2.*pi) - 0.5*log( (s2+sigm2_i) )  + ( -0.5*(y(i1)-myy_i).^2./(s2+sigm2_i) );
% 
% end

function [logM_0, m_1, sigm2hati1] = lik_huber_tiltedMoments(lik, y, i1, sigma2_i, myy_i, z)
%LIK_LAPLACE_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%  Description
%    [M_0, M_1, M2] = LIK_LAPLACE_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY, Z) takes a likelihood structure LIK, observations
%    Y, index I and cavity variance S2 and mean MYY. Returns 
%    the zeroth moment M_0, mean M_1 and variance M_2 of the 
%    posterior marginal (see Rasmussen and Williams (2006): 
%    Gaussian processes for Machine Learning, page 55). This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods.
%
%  See also
%    GPEP_E
  
  yy = y(i1);
  sigma2 = lik.sigma2;
  sigma=sqrt(sigma2);
%   weights=lik.weights;
%   b=lik.b;
%   epsilon=lik.epsilon;
  logM_0=zeros(size(yy));
  m_1=zeros(size(yy));
  sigm2hati1=zeros(size(yy));
  
  for i=1:length(i1)
    if isscalar(sigma2_i)
      sigma2ii = sigma2_i;
    else
      sigma2ii = sigma2_i(i);
    end
    
    % get a function handle of an unnormalized tilted distribution
    % (likelihood * cavity = Quantile-GP * Gaussian)
    % and useful integration limits
    [tf,minf,maxf]=init_huber_norm(yy(i),myy_i(i),sigma2ii,sigma);
    
    % Integrate with quadrature
    RTOL = 1.e-6;
    ATOL = 1.e-10;
    [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
    if isnan(m_0)
      logM_0=NaN;
      return
    end
    sigm2hati1(i) = m_2 - m_1(i).^2;
    
    % If the second central moment is less than cavity variance
    % integrate more precisely. Theoretically for log-concave
    % likelihood should be sigm2hati1 < sigm2_i.
    if sigm2hati1(i) >= sigma2ii
      ATOL = ATOL.^2;
      RTOL = RTOL.^2;
      [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
      sigm2hati1(i) = m_2 - m_1(i).^2;
      if sigm2hati1(i) >= sigma2ii
        warning('lik_huber_tilted_moments: sigm2hati1 >= sigm2_i');
      end
    end
    logM_0(i) = log(m_0);
  end
end


% function [f, a] = lik_huber_optimizef(gp, y, K, Lav, K_fu)
% %LIK_T_OPTIMIZEF  function to optimize the latent variables
% %                 with EM algorithm
% %
% %  Description:
% %    [F, A] = LIK_T_OPTIMIZEF(GP, Y, K, Lav, K_fu) Takes Gaussian
% %    process structure GP, observations Y and the covariance
% %    matrix K. Solves the posterior mode of F using EM algorithm
% %    and evaluates A = (K + W)\Y as a sideproduct. Lav and K_fu
% %    are needed for sparse approximations. For details, see
% %    Vanhatalo, Jylänki and Vehtari (2009): Gaussian process
% %    regression with Student-t likelihood. This subfunction is 
% %    needed when using lik_specific optimization method for mode 
% %    finding in Laplace algorithm.
% %
%   
%   iter = 1;
%   sigma2 = gp.lik.sigma2;
% %  if sigma2==0
% %    f=NaN;a=NaN;
% %    return
% %  end
%  sigma=sqrt(sigma2);
%   n = length(y);
%   
%   switch gp.type
%     case 'FULL'            
%       iV = ones(n,1)./sigma;
%       siV = sqrt(iV);
%       B = eye(n) + siV*siV'.*K;
%       [L,notpositivedefinite] = chol(B);
%       if notpositivedefinite
%         f=NaN;a=NaN;
%         return
%       end
%       B=B';
%       b = iV.*y;
%       a = b - siV.*(L'\(L\(siV.*(K*b))));
%       f = K*a;
%       while iter < 200
%         fold = f;               
%         bbbb=0.5;
%         epsilon=0.45;
%         rs=(f-y)./sigma;
%         rho=@(rs) bbbb^2*(sqrt(1+(rs./bbbb).^(2))-1);
%         
%         iV = ((1-epsilon)./(sqrt(2*pi).*sigma)).*exp(-rho(rs)).*(-rho(rs)).*(0.5)./((sqrt(1+(rs./bbbb).^(2))-1)).*2.*rs./bbbb.*(1/sigma);
%         siV = sqrt(iV);
%         B = eye(n) + siV*siV'.*K;
%         L = chol(B)';
%         b = iV.*y;
%         ws=warning('off','MATLAB:nearlySingularMatrix');
%         a = b - siV.*(L'\(L\(siV.*(K*b))));
%         warning(ws);
%         f = K*a;
%         
%         if max(abs(f-fold)) < 1e-8
%           break
%         end
%         iter = iter + 1;
%       end
%     case 'FIC'
%       K_uu = K;
%       
%       Luu = chol(K_uu)';
%       B=Luu\(K_fu');       % u x f
% 
%       K = diag(Lav) + B'*B;
%       
%       iV = ones(n,1)./sigma2;
%       siV = sqrt(iV);
%       B = eye(n) + siV*siV'.*K;
%       L = chol(B)';
%       b = iV.*y;
%       a = b - siV.*(L'\(L\(siV.*(K*b))));
%       f = K*a;
%       while iter < 200
%         fold = f;                
%         iV = (nu+1) ./ (nu.*sigma2 + (y-f).^2);
%         siV = sqrt(iV);
%         B = eye(n) + siV*siV'.*K;
%         L = chol(B)';
%         b = iV.*y;
%         a = b - siV.*(L'\(L\(siV.*(K*b))));
%         f = K*a;
%         
%         if max(abs(f-fold)) < 1e-8
%           break
%         end
%         iter = iter + 1;
%       end
%   end
%   
% end











function [g_i] = lik_huber_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)   %not sure 

%LIK_NEGBIN_SITEDERIV  Evaluate the expectation of the gradient
%                      of the log likelihood term with respect
%                      to the likelihood parameters for EP 
%
%  Description [M_0, M_1, M2] =
%    LIK_NEGBIN_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
%    likelihood structure LIK, incedence counts Y, expected
%    counts Z, index I and cavity variance S2 and mean MYY. 
%    Returns E_f [d log p(y_i|f_i) /d a], where a is the
%    likelihood parameter and the expectation is over the
%    marginal posterior. This term is needed when evaluating the
%    gradients of the marginal likelihood estimate Z_EP with
%    respect to the likelihood parameters (see Seeger (2008):
%    Expectation propagation for exponential families). This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods and there are likelihood parameters.
%
%  See also
%    GPEP_G

% tau = 1/s2 + 1/sigm2_i;
% w = (y(i1)/s2 + myy_i/sigm2_i)/tau;
% %Zi = 1/( sqrt(2*pi*(s2+sigm2_i)) )*exp( -0.5*(y(i1)-myy_i)^2/(s2+sigm2_i) );
% 
% %g_i = 0.5*( (1/tau + w.^2 -2*w*y(i1) + y(i1).^2 ) /s2 - 1)/s2 * s2;
% g_i = 0.5*( (1/tau + w.^2 -2*w*y(i1) + y(i1).^2 ) /s2 - 1);
sigma=sqrt(lik.sigma2);
b=0.5;
yy = y(i1);

  % get a function handle of an unnormalized tilted distribution 
  % (likelihood * cavity = Quantile-GP * Gaussian)
  % and useful integration limits
  [tf,minf,maxf]=init_huber_norm(yy,myy_i,sigm2_i,sigma);
  % additionally get function handle for the derivative
  td = @deriv;
  
  % Integrate with quadgk
  [m_0, fhncnt] = quadgk(tf, minf, maxf);
  [g_i, fhncnt] = quadgk(@(f) td(f).*tf(f)./m_0, minf, maxf);
  

  function g = deriv(f)
   epsilon=0.45;
   weights=ones(length(y),1);
   weights=lik.weights;

    g = -sqrt(2*pi)./(sigma*(1-epsilon)) +(1./(2* sqrt(((yy-f).^2./(weights.^2.*sigma^2))+1))).*((yy-f)./sigma).^2;
    
  end
end


function [df,minf,maxf] = init_huber_norm(yy,myy_i,sigm2_i,sigma)
%INIT_LAPLACE_NORM
%
%  Description
%    Return function handle to a function evaluating
%    Laplace * Gaussian which is used for evaluating
%    (likelihood * cavity) or (likelihood * posterior) Return
%    also useful limits for integration. This is private function
%    for lik_laplace. This subfunction is needed by subfunctions
%    tiltedMoments, siteDeriv and predy.
%  
%  See also
%    LIK_LAPLACE_TILTEDMOMENTS, LIK_LAPLACE_SITEDERIV,
%    LIK_LAPLACE_PREDY

%   s2=lik.sigma2;
%   sigma= sqrt(s2);
%   weights=lik.weights;
%   epsilon=lik.epsilon;
%   b=lik.b;
% avoid repetitive evaluation of constant part
epsilon=0.45;
b=0.5;
  ldconst = -log(2*sigma)+log(1-epsilon) ...
            - log(sigm2_i)/2 - log(2*pi);
  % Create function handle for the function to be integrated
  df = @huber_norm;
  % use log to avoid underflow, and derivates for faster search
  ld = @log_huber_norm;
  ldg = @log_huber_norm_g;
%   ldg2 = @log_laplace_norm_g2;

  % Set the limits for integration
  % Quantile-GP likelihood is log-concave so the laplace_norm
  % function is unimodal, which makes things easier
  if yy==0
    % with yy==0, the mode of the likelihood is not defined
    % use the mode of the Gaussian (cavity or posterior) as a first guess
    modef = myy_i;
  else
    % use precision weighted mean of the Gaussian approximation
    % of the Quantile-GP likelihood and Gaussian
    modef = (myy_i/sigm2_i + yy/sigma)/(1/sigm2_i + 1/sigma);
  end
  % find the mode of the integrand using Newton iterations
  % few iterations is enough, since the first guess in the right direction
  niter=8;       % number of Newton iterations 
  
  minf=modef-6*sigm2_i;
  while ldg(minf) < 0
    minf=minf-2*sigm2_i;
  end
  maxf=modef+6*sigm2_i;
  while ldg(maxf) > 0
    maxf=maxf+2*sigm2_i;
  end
  for ni=1:niter
%     h=ldg2(modef);
    modef=0.5*(minf+maxf);
    if ldg(modef) < 0
      maxf=modef;
    else
      minf=modef;
    end
  end
  % integrand limits based on Gaussian approximation at mode
  minf=modef-6*sqrt(sigm2_i);
  maxf=modef+6*sqrt(sigm2_i);
  modeld=ld(modef);
  iter=0;
  % check that density at end points is low enough
  lddiff=20; % min difference in log-density between mode and end-points
  minld=ld(minf);
  step=1;
  while minld>(modeld-lddiff)
    minf=minf-step*sqrt(sigm2_i);
    minld=ld(minf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_huber -> init_huber_norm: ' ...
             'integration interval minimun not found ' ...
             'even after looking hard!'])
    end
  end
  maxld=ld(maxf);
  iter=0;
  step=1;
  while maxld>(modeld-lddiff)
    maxf=maxf+step*sqrt(sigm2_i);
    maxld=ld(maxf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_huber -> init_huber_norm: ' ...
             'integration interval maximun not found ' ...
             'even after looking hard!'])
    end
  end
  
  function integrand = huber_norm(f)

%       c1=1+5/(length(f(:,1))-length(f(1,:)));
%       s1 = 1.4826*(c1)*median(abs(yy-f));
      b=0.5;;
      weights=ones(length(yy),1);
%        weights=lik.weights;
      epsilon=0.45;
  % Huber * Gaussian
    integrand = exp(ldconst ...
                   - b^2.*(sqrt(1+((yy-f)./(weights.*b.*sigma)).^2)-1) ...
                    -0.5*(f-myy_i).^2./sigm2_i);
  end
  
  function log_int = log_huber_norm(f)

%       c1=1+5/(length(f(:,1))-length(f(1,:)));
%       s1 = 1.4826*(c1)*median(abs(yy-f));
      b=0.5;;
      weights=ones(length(yy),1);
%        weights=lik.weights;
      epsilon=0.45;

  % log(Huber * Gaussian)
  % log_huber_norm is used to avoid underflow when searching
  % integration interval
    log_int = ldconst...
             -b^2.*(sqrt(1+((yy-f)./(weights.*sigma.*b)).^2)-1) ...
                    -0.5*(f-myy_i).^2./sigm2_i;
  end
  
  function g = log_huber_norm_g(f)
%       c1=1+5/(length(f(:,1))-length(f(1,:)));
%       s1 = 1.4826*(c1)*median(abs(yy-f));
      b=0.5;;
      weights=ones(length(yy),1);
%        weights=lik.weights;
  % d/df log(Laplace * Gaussian)
  % derivative of log_huber_norm
%     g = (yy-f)./(sqrt(1+((yy-f)./(weights.*b)).^(2)).*sigma.^(2).*weights.^(2)) ...
%         + (myy_i - f)./sigm2_i;

     g = -(f - yy)./(sigma^2.*weights.^2.*((f - yy).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2)) + (myy_i - f)./sigm2_i;


  end
  
  
end


function mu = lik_huber_invlink(lik, f, z)
%LIK_LAPLACE_INVLINK  Returns values of inverse link function
%             
%  Description 
%    MU = LIK_LAPLACE_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values MU of inverse link function.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_LAPLACE_LL, LIK_LAPLACE_PREDY
  
  mu = f;
end



function reclik = lik_huber_recappend(reclik, ri, lik)
%RECAPPEND  Record append
%
%  Description
%    RECLIK = LIK_GAUSSIAN_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood function record structure RECLIK, record index RI
%    and likelihood function structure LIK with the current MCMC
%    samples of the parameters. Returns RECLIK which contains all
%    the old samples and the current samples from LIK. This 
%    subfunction is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC and GP_MC -> RECAPPEND

  if nargin == 2
    % Initialize the record
    reclik.type = 'Huber';
    
    % Initialize the parameters
    reclik.sigma2 = []; 
    reclik.n = []; 
    
    % Set the function handles
    reclik.fh.pak = @lik_huber_pak;
    reclik.fh.unpak = @lik_huber_unpak;
    reclik.fh.lp = @lik_huber_lp;
    reclik.fh.lpg = @lik_huber_lpg;
    reclik.fh.ll= @lik_huber_ll;
    reclik.fh.llg = @lik_huber_llg;
    reclik.fh.llg2 = @lik_huber_llg2;
    reclik.fh.llg3 = @lik_huber_llg3;
    reclik.fh.tiltedMoments = @lik_huber_tiltedMoments;
    reclik.fh.siteDeriv = @lik_huber_siteDeriv;
    reclik.fh.lik_huber_invlink=@lik_huber_invlink;
%     reclik.fh.lik_huber_optimizef=@lik_huber_optimizef;
%     reclik.fh.cfg = @lik_huber_cfg;
%     reclik.fh.trcov  = @lik_huber_trcov;
%     reclik.fh.trvar  = @lik_huber_trvar;
    reclik.fh.predy = @lik_huber_predy;
    reclik.fh.recappend = @lik_huber_recappend;     
    reclik.p=[];
    reclik.p.sigma2=[];
    if ~isempty(ri.p.sigma2)
      reclik.p.sigma2 = ri.p.sigma2;
    end
  else
    % Append to the record
    likp = lik.p;

    % record sigma2
    reclik.sigma2(ri,:)=lik.sigma2;
    if isfield(likp,'sigma2') && ~isempty(likp.sigma2)
      reclik.p.sigma2 = likp.sigma2.fh.recappend(reclik.p.sigma2, ri, likp.sigma2);
    end
    % record n if given
    if isfield(lik,'n') && ~isempty(lik.n)
      reclik.n(ri,:)=lik.n(:)';
    end
  end
end
