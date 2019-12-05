%This is a script to create a 10 armed bandit testbed using Greedy v/s
%non-greedy algorithms. 
% No. of runs= 2000
% Time steps per run: 1000
% q*(a)= N(0,1)
% R(t) for At= N(q*(At),1)
% e= 0.1 and 0.01
clc;
close all;
%Define q*(a)
for n=1:2000
    A= zeros(10,1000);% Action register
    R=zeros(10,1000); %Reward register
    Q= zeros(10,1000); %Value register
    for i=1:10
        a(i)= normrnd(0,1);        
    end
    %Inital
    f= randi(10);
    A(f,1)=1;
    R(f,1)= normrnd(a(f),1);
    
    
    %After initial 
    for t=2:1000
        %Determine Action Value estimate using sample average method
        %Reward matrix * 1(Action matrix) (hadamard product)
        for i=1:10
            if nnz(A(i,:))-A(i,t)~=0 
            Q(i,t)= (sum(R(i,:))-R(i,t))/ ( nnz(A(i,:))-A(i,t)); %Sample Average till t-1
            end
            if nnz(A(i,:))-A(i,t)==0 
                Q(i,t)=0;
            end
        end
          
        [M,I]= max(Q(:,(t-1)));
        %Multiple greedy actions
        K= find(Q(:,(t-1))==M);
        r= randi(length(K));
        k= K(r);
        
        A(k,t)=1;%Action Register
        R(k,t)= normrnd(a(k),1); %Reward register
    end
    
%Total reward
RewardG(n,:)=sum(R);
end
%Average and Plot
for t=1:1000
    Pas(t)= mean(RewardG(:,t));
end
t=1:1000;
plot(t,Pas)