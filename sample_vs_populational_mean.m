%This is the code to plot the sample means (mean_s) vs the populational means (mean_u) using
%different sampel size.
"""
Created on Tue Sep 27 16:40:08 2020
last modified: April 19, 2022
"""

clear all; close all; clc

data=load('datas/all_data_for_training_simple.csv');
min_val=[0.01, 0.01, 0.01];
max_val=[1000, 1000, 200];
params=data(:,1:3).*(max_val-min_val)+min_val;
kon=params(:,1);
koff=params(:,2);
v=params(:,3);

mean_u=v.*kon./(kon+koff);
sample_size=10;
samples=data(:,4:(4+sample_size-1));
mean_s=mean(samples,2);
xmax=max([max(mean_u),max(mean_s)]);

figure(1)
scatter(mean_u,mean_s,'filled')
xlim([0,200])
ylim([0,200])
xlabel('u')
ylabel('s')
axis square
set(gca,'fontSize',20, 'LineWidth',3)

figure(2)
scatter(mean_u,mean_s,'filled')
ylim([55,65])
%xlim([50,70])
xlabel('u')
ylabel('s')
axis square
set(gca,'fontSize',20, 'LineWidth',3)

dist_data=load('datas/full_distributions_simple.csv');
idx=randperm(10000);
%sample visualization
figure(3)
for i=1:9
subplot(3,3,i)
hold on;
sample1=samples(idx(i),:);
param1=data(idx(i),1:3);
idx_dist=find(dist_data(:,1)==param1(1) & dist_data(:,2)==param1(2) & dist_data(:,3)==param1(3));
dist=dist_data(idx_dist,4:end);
plot(0:500,dist,'b','LineWidth',2);
for s=1:3
    plot([sample1(s),sample1(s)],[0,max(dist)],'g','LineWidth',2)
end
    
xlim([0,200])

end


