function[K,R]=kernel_cal(X1,X2,l,tau)
m1=length(X1(:,1));
m2=length(X2(:,1));
K=ones(m1,m2);
R=ones(m1,m2);

for i=1:m1
    for j=1:m2
        
            R(i,j)=exp(-0.5*sum(((X1(i,:)-X2(j,:)).^2./(l'.^2))));
            K(i,j)=tau^2*R(i,j);
%              sum=sum+sin(1+pi*abs((X1(i,k)-X2(j,k))/(l(k)))); %2
%                sum=sum+abs(((X1(i,k)-X2(j,k)))/(l(k))); %3
%                sum= sum+sqrt(3)*(X1(i,k)-X2(j,k))/l(k);  %4

        
    end 
end
end