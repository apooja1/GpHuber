%Function to calculate the projection statistics (PS)
%Reference:
%[1] Mili, L.; Cheniae, M.G.; Vichare, N.S.; Rousseeuw, P.J., "Robust 
%    state estimation based on projection statistics [of power systems]," 
%    Power Systems, IEEE Transactions on , vol.11, no.2, pp.1118,1127, 
%    May 1996.
%
function [P,PS] = projectionstatistics(H)
% Correlation coefficient
% disp('Inasmuch n-->inf the Covariance matrix approximates');
% disp('the Identity matrix. Correlation coefficient:');
% R = corrcoef(h')
[m,n]=size(H);
M=median(H);                                            
u=zeros(m,n);
v=zeros(m,n);
z=zeros(m,1);
P=zeros(m,m);
for kk=1:m
    u(kk,:)=H(kk,:)-M;                                  
    v(kk,:)=u(kk,:)/norm(u(kk,:));                      
    for ii=1:m
        z(ii,:)=dot(H(ii,:)',v(kk,:));                  
    end
    zmed=median(z);                                     
    MAD=1.4826*(1+(15/(m)))*median(abs(z-zmed));         
    for ii=1:m                                          
        P(kk,ii)=abs(z(ii)-zmed)/MAD;
    end
end
PS=max(P);                                              % Step 9: Calculate the projection statistics