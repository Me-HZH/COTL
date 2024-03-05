function  [f1,precision,recall,G_mean] =  Experiment(source_project, target_project)
% 
%--------------------------------------------------------------------------
% Input:
%      source_project: source dataset_name
%      target_project: target dataset_name
%--------------------------------------------------------------------------

%load dataset
load(sprintf('E:/Matlab_project/COTL/data/%s', source_project));
source_data = data(:,:);
ID = ID_old;

load(sprintf('E:/Matlab_project/COTL/data/%s', target_project));
target_data = data(ID_old,:);

rate = 0.3;
m = size(target_data,1);
num = int64(m*rate);
ID_new=[];
for i=1:20,
    ID_new = [ID_new; 1:(m - num)];
end

% options
options.C = 5;
options.k = labels_num; 
options.Tw = 30;
options.dim = 10;
options.mu = 0.1;
options.alpha = 0.1;
options.lambda = 1;
m = length(ID_new);
options.beta1 = sqrt(m)/(sqrt(m)+sqrt(log(2)));
options.beta2 = sqrt(m)/(sqrt(m)+sqrt(log(2)));

%% run experiments:
[h,Pt,X,Y, mean_Xt, num, mean_kt, num_kt] = source_classifier(source_data,target_data,ID,rate,options);
NUM_t = double(num); 
NUM_kt = double(num_kt); 



for i=1:20,
    ID = ID_new(i, :);
    [score,precision,recall,G_mean] = COTL(Y,Pt,X,mean_Xt,NUM_t,mean_kt,NUM_kt,options,ID,h);
    all_score(i) = score;
    all_precision(i) = precision;
    all_recall(i) = recall;
    all_G_mean(i) = G_mean;
end
precision = mean(all_precision);
recall = mean(all_recall);
f1 = mean(all_score);
G_mean = mean(all_G_mean);
