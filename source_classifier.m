function [ h, Pt, X, Y, mean_Xt, num, mean_kt, num_kt] = source_classifier(source_data,target_data,ID,rate,options)

data = [source_data;target_data];
[n,d] = size(data);
m = size(target_data,1);
Y=data(1:n,1);
Y=full(Y);
X = data(1:n,2:d);
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);
X = X';
X = X*diag(sparse(1./sqrt(sum(X.^2))));
X = X';
X = zscore(X);
X = normr(X)';
X = X';

num = int64(m*rate);
X2 = X(n-m+1:n-m+num,:);  %online data
Y2 = Y(n-m+1:n-m+num);

mean_Xt = mean(X(n-m+1:n-m+num,:));  %mean of unlabeled target data

num_ks = zeros(options.k,1);
sum_ks = zeros(options.k,d-1);

for i = 1:n-m
    num_ks(Y(i)) = num_ks(Y(i)) + 1;
    sum_ks(Y(i),:) = sum_ks(Y(i),:) + X(i,:);
end


num_kt = zeros(options.k,1);
sum_kt = zeros(options.k,d-1);
for i = n-m+1:n-m+num
    num_kt(Y(i)) = num_kt(Y(i)) + 1;
    sum_kt(Y(i),:) = sum_kt(Y(i),:) + X(i,:);
end

for i = 1:options.k
    mean_kt(i,:) = sum_kt(i,:)./num_kt(i);
end

% CDSPP
[P] = CDSPP(X(1:n-m,:),X2,Y(1:n-m)',Y2',options);
Ps = P(1:size(X(1:n-m,:),2),:);
Pt = P(size(X(1:n-m,:),2)+1:end,:);
Zs = X(1:n-m,:)*Ps;
[h] = avePA1_K_M(Y(1:n-m), Zs, options, ID);

X = X(n-m+num+1:n,:);
Y = Y(n-m+num+1:n);
end

