% This script is used to obtain the results of the GP-Huber model on Neal dataset for rebuttal.
%% Neal data
% Model:
% f = 0.3 + 0.4x + 0.5sin(2.7x) + 1.1/(1+x^2).

close all 
clc 
clear all
rng(5)
Results=zeros();

%% importing the data 
% sheet = 1;
opts = detectImportOptions('Energy_data_assymetric.xlsx');
opts.Sheet = 'Train';
Train = readmatrix('Energy_data_assymetric.xlsx',opts);

% Train=readmatrix('Yacht_data_focused1.xlsx',sheet);
x=Train(:,1:end); y=Train(:,end-1);
% sheet = 2;
opts.Sheet = 'Test';
% Test=xlsread('Yacht_data_focused1.xlsx',sheet);
Test=readmatrix('Energy_data_assymetric.xlsx',opts);
xt=Test(:,1:end); yt=Test(:,end-1);
% sheet=3;
opts.Sheet = 'Results';
Re=readmatrix('Energy_data_assymetric.xlsx',opts);
% Re=[0.376892163;	0.216696926;	1.193936434];


% writematrix([xt yt],'Neal_data.xls','Sheet',2)
% Energy_data_assymetric
% Yacht_data_focused
% Yacht_data_assymetric

[n, nin] = size(x); 


% Laying priors
pl = prior_t();
pm = prior_sqrtunif();
pn = prior_logunif();

%% Comparison models 

%% Model 1
% ========================================
% MCMC approach with scale mixture noise model (~=Student-t)
% Here we sample all the variables 
%     (lengthScale, magnSigma, sigma(noise-t) and nu)
% ========================================
disp(['Scale mixture Gaussian (~=Student-t) noise model                ';...
      'using MCMC integration over the latent values and parameters    '])

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.001, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);  %covariance structure
% Here, set own Sigma2 for every data point
lik = lik_gaussiansmt('ndata', n, 'sigma2', repmat(0.2^2,n,1), ...
                      'nu_prior', prior_logunif());    %Student-t scale mixture model
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9);   % positive jitter perhaps the nugget element

% Sample
[r,g,opt]=gp_mc(gp, x, y, 'nsamples', 300, 'display', 20); 

% thin the record
rr = thin(r,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft=sqrt(Varft);

model=1;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

% Plot the network outputs as '.', and underlying mean with '--'
%   figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
%  set(gcf,"Color",'w');
%  set(gca,'FontSize',15,'FontWeight','bold')
% ax=gca;
% exportgraphics(ax,'Neal_SCtMCMC.png')
% axis on;
S2 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))



%% Model 2
disp(['Student-t noise model with nu= 4 and using MCMC integration';...
      'over the latent values and parameters                      '])

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.01, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
pn = prior_logunif();
lik = lik_t('nu', 4, 'nu_prior', [], 'sigma2', 0.2^2, 'sigma2_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-4, ...
             'latent_method', 'MCMC');
f=gp_pred(gp,x,y,x);
gp=gp_set(gp, 'latent_opt', struct('f', f));

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, 'display', 20);
rr = thin(rgp,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft = sqrt(Varft);

% Plot the network outputs as '.', and underlying mean with '--'
% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% %   legend boxoff
% xlim([-2.7 4.8])
% set(gcf,"Color",'w');
% set(gca,'FontSize',15,'FontWeight','bold')
% ax=gca;
% exportgraphics(ax,'Neal_SCt4MCMC.png')
% axis on;

S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

model=2;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
%% Model 3
disp(['Student-t noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])

gpcf = gpcf_sexp('lengthScale', 10, 'magnSigma2', 0.001, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
pn = prior_logunif();
lik = lik_t('nu', 4, 'nu_prior', pn, ...
            'sigma2', 0.001, 'sigma2_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-6, ...
            'latent_method', 'Laplace');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt);

% Predictions to test points
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
% set(gcf,"Color",'w');
% set(gca,'FontSize',15,'FontWeight','bold')
% ax=gca;
% exportgraphics(ax,'Neal_tLA.png')
% axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)
model=3;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt)
 

%% Model 4 in the paper
% ========================================
% MCMC approach with Huber observation noise model
% ========================================
weights=zeros();
 b=1.5;
 X=x;
 c1=1+5/(length(X(:,1))-length(X(1,:)));
 H=[ones(length(X(:,1)),1) X];
%  H=[X y];
 [P,PS]=projectionstatistics(H);
 [m n]=size(X);
  for i=1:m
   niu=sum(H(i,:)~=0);
   cuttoff_PS(i,1)=chi2inv(0.975,niu);
   weights(i,1)=min(1,(cuttoff_PS(i,1)/PS(i)^2)); %% downweight the outliers or leverage points
  end

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.02, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_huber('sigma2', 0.8^2, 'sigma2_prior', pn,'weights',weights);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
             'latent_method', 'MCMC');
f=gp_pred(gp,x,y,x);
gp=gp_set(gp, 'latent_opt', struct('f', f));

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, 'display', 20);
rr = thin(rgp,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft = sqrt(Varft);

model=4;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
%   set(gcf,"Color",'w');
%  set(gca,'FontSize',15,'FontWeight','bold')
% saveas(gcf,'Neal_HuberMCMC','epsc')
% axis on;
% S2 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
%              mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%% Model 5 in the paper
disp(['Huber     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters        '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 2, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
% lik = lik_huber('sigma2', 0.2^2, 'sigma2_prior', pn,'weights',weights,'b',0.5,'epsilon',0.45);
lik = lik_huber('sigma2', 0.8^2, 'sigma2_prior', pn,'weights',weights);
% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
            'latent_method', 'Laplace');

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt,'loss','loo');
% gp=gp_optim(gp,x,y,'opt',opt);
% Predictions to test points
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}+-2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
%   set(gcf,"Color",'w');
%  set(gca,'FontSize',15,'FontWeight','bold')
% saveas(gcf,'Neal_HuberLA','epsc')
% axis on;
S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

model=5;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

%% % Gaussian model 
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.001, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);
lik = lik_gaussian('sigma2', 0.2^2, 'sigma2_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9)

% --- MAP estimate ---
disp('Gaussian noise model and MAP estimate for parameters')

% Set the options for the optimization
opt=optimset('TolFun',1e-3,'TolX',1e-3);
% Optimize with the scaled conjugate gradient method
gp=gp_optim(gp,x,y,'opt',opt);

% Prediction
[Eft, Varft, lpyt, Eyt, Varyt] = gp_pred(gp, x, y, xt, 'yt', ones(size(xt)));
std_ft = sqrt(Varft);

model=6;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
% Plot the prediction and data
% plot the training data with dots and the underlying 
% mean of it as a line

% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
%   set(gcf,"Color",'w');
%  set(gca,'FontSize',15,'FontWeight','bold')
% ax=gca;
% exportgraphics(ax,'Neal_GP.png')
% axis on;



S1 = sprintf('length-scale: %.3f, magnSigma2: %.3f  \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)


%% Model 7
% Laplace Observation model with MCMC

gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.001, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);

% Create the likelihood structure
lik = lik_laplace('scale', 2, 'scale_prior', pn);

% ... Finally create the GP structure
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-9, ...
             'latent_method', 'MCMC');
f=gp_pred(gp,x,y,x);
gp=gp_set(gp, 'latent_opt', struct('f', f));

% Sample 
[rgp,g,opt]=gp_mc(gp, x, y, 'nsamples', 400, 'display', 20);
rr = thin(rgp,100,2);

% make predictions for test set
[Eft, Varft] = gp_pred(rr,x,y,xt);
std_ft = sqrt(Varft);

model=7;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);

% Plot the network outputs as '.', and underlying mean with '--'
% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
% set(gcf,"Color",'w');
% set(gca,'FontSize',15,'FontWeight','bold')
% ax=gca;
% exportgraphics(ax,'Neal_LaplaceMCMC.png')
% axis on;
S5 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', mean(rr.cf{1}.lengthScale), mean(rr.cf{1}.magnSigma2))

%%  Model 8
disp(['Laplace     noise model using Laplace integration over the '; ...
      'latent values and MAP estimate for the parameters          '])


% Create the likelihood structure
gpcf = gpcf_sexp('lengthScale', 1, 'magnSigma2', 0.001, ...
                  'lengthScale_prior', pl, 'magnSigma2_prior', pm);


lik = lik_laplace('scale', 0.4, 'scale_prior', pn);
gp = gp_set('lik', lik, 'cf', gpcf, 'jitterSigma2', 1e-5, ...
            'latent_method', 'EP');

opt=optimset('TolFun',1e-3,'TolX',1e-3);
gp=gp_optim(gp,x,y,'opt',opt);
[Eft, Varft] = gp_pred(gp, x, y, xt);
std_ft = sqrt(Varft);

% Plot the prediction and data
% figure
%   mu=Eft; s2=std_ft;
%   f = [mu+2*(s2); flipdim(mu-2*(s2),1.5)];
%   fill([xt; flipdim(xt,1)], f,  [7 7 7]/8,'FaceAlpha',0.8,'EdgeColor', [7 7 7]/8,'LineStyle','--')
%   hold on; 
%   plot(xt,yt,'color',[0 0 0],LineWidth=1.5);
%   plot(xt, mu,'color','r'); 
%   plot(x, y, 'k.',LineWidth=1.5);
%   set(legend('$\hat{\textrm{f}}\pm 2\textrm{std}(\hat{\textrm{f}})$','real f','$$\hat{{\textrm{f}}}$$','y','Location', 'Best'),'Interpreter','Latex','FontSize', 15,'FontWeight','bold')
% 
% %   legend boxoff
% xlim([-2.7 4.8])
% set(gcf,"Color",'w');
% set(gca,'FontSize',15,'FontWeight','bold')
% ax=gca;
% exportgraphics(ax,'Neal_LaplaceEP.png')
% axis on;

S3 = sprintf('length-scale: %.3f, magnSigma2: %.3f \n', ...
             gp.cf{1}.lengthScale, gp.cf{1}.magnSigma2)

model=8;
Results(model,1) = 1 - var(Eft-yt)/var(yt);
Results(model,2) = sqrt(mean((Eft-yt).^2));
Results(model,3) = mean(abs(Eft-yt));
Results(model,4) = NLP(Eft,Varft, yt);
%%
Results= (Results(:,2:end))'; 
Results=[Results Re];
Results= round(Results, 2);
matrix2latex(Results, 'out.tex')

