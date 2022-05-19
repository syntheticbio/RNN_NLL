%This code is to generate training data for the NN to learn the probability distributions.
"""
Created on Tue Sep 27 16:40:08 2020
last modified: April 19, 2022
"""
%data structure:
%1-4 input parameters, k_on, k_off, v,\delta
%5-7 3 observations randomly drawn from probability distributions

%k_on: 0.01-1000
%k_off: 0.01-1000
%v:0.01-300
%\delta: 1.0 (all other parameters are normalized with \delta)
clear all; close all; clc

taskID=getenv('SLURM_ARRAY_TASK_ID');
%# iterations
offset = 1;
%reinitialize the random number generator based on the time and the process-ID of the matlab to make sure every cluster node run differently
rng('shuffle'); % seed with the current time
rngState = rng; % current state of rng

%%% deltaSeed can be any data unique to the process, 
%%% including position in the process queue
deltaSeed = uint32(feature('getpid'));

seed = rngState.Seed + deltaSeed;
rng(seed); % set the rng to use the modified seed,which would combine the current time with the process-ID of the matlab instance to generate the seed.


%%
%randomly generate parameter combinations
num_sample=2;
num_par=3;
dim=500; % number os observations of each distribution, if 3 observations are used for the training, then the first 2 observations are used.
rands=rand(num_sample,num_par);
min_val=[0.01, 0.01, 0.01];
max_val=[1000, 1000, 200];
params=rands.*(max_val-min_val)+min_val;
mRNA_array=0:500;
m_length=length(mRNA_array);
all_data=zeros(num_sample,num_par+dim);
all_data(:,1:num_par)=rands;
probabilities=zeros(num_sample,m_length);
syms n
for i=1:num_sample
kon=params(i,1);
koff=params(i,2);
v=params(i,3);

digitsOld = digits(256);
a1=vpa(kon);
b1=vpa(kon+koff);
c1=vpa(v);
num_temp=(c1^n)/factorial(n)*real(hypergeom(n+a1,n+b1,-c1))*gamma(n+a1)/gamma(a1)*gamma(b1)/gamma(n+b1);
for j=0:(m_length-1)
    probabilities(i,j+1)=subs(num_temp,n,j);
end
digits(digitsOld)

%randomly drawn 3 samples from this distributions.
cdfp=cumtrapz(mRNA_array,probabilities(i,:)); %% computing the cumulative distribution function for input pdf

% finding the parts of cdf parallel to the X axis
ind=[true not(diff(cdfp)==0)];
% and cut out the parts
cdfp=cdfp(ind);
pxi=mRNA_array(ind);

% generating the uniform distributed random numbers
uniformDistNum=rand(1,dim);
% and distributing the numbers using cdf from input pdf
select_samples=interp1(cdfp,pxi,uniformDistNum,'linear');
%plot(mRNA_array,probabilities(i,:))
%histogram(select_samples)
all_data(i,(num_par+1):end)=select_samples;
end



file_name= fullfile(pwd, sprintf('datas/datas_%s.mat',datastr(now,'dd_mm_yyyy_AM'),taskID)); 
save(file_name,'rands','probabilities','all_data');


