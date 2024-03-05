function [score,precision,recall,G_mean] = COTL(Y, Pt, X,mean_Xt,NUM_t,mean_kt,NUM_kt, options, id_list,classifiers)

%% initialize parameters
beta = options.beta1;
C = options.C; 
k = options.k;
mu = options.mu;
Tw = options.Tw;
temp = 0;
MEAN_Xt0 = mean_Xt;
MEAN_kt0 = mean_kt;
u_t = [];
v_t = [];
X2t = [];
Ws = classifiers.w;
Wt = zeros(1,options.dim);
u_t = [u_t, 1/2];
v_t = [v_t, 1/2];
p_s = u_t / (sum(u_t, 2) + sum(v_t,2));
p_t = v_t / (sum(u_t, 2) + sum(v_t,2));
ID = id_list;
y_p = [];
y = [];

for t = 1:length(ID),
    id = ID(t);
    id_new = id;
    x_t = X(id_new, :);
    y_t = Y(id_new);
    x_t = x_t*Pt;
    x_t = x_t*(1/sqrt(sum(x_t.^2,2)));
    F_s = Ws*x_t'; 
    F_t = Wt*x_t';
    p_s = u_t / (sum(u_t, 2) + sum(v_t,2));
    p_t = v_t / (sum(u_t, 2) + sum(v_t,2));
    u_t = p_s;
    v_t = p_t;
    F = 0;
    F = F + p_s*F_s + p_t*F_t;
    if (y_t==1)
        yt = -1;
    end
    if (y_t==2)
        yt = 1;
    end

    hat_y_t = sign(F);
    if (hat_y_t==0)
        hat_y_t=1;
    end

    y_p = [y_p hat_y_t];
    y = [y y_t];
    hat_y_t1 = sign(F_s);
    hat_y_t2 = sign(F_t);
    if (hat_y_t1==0)
        hat_y_t1=1;
    end
    if (hat_y_t2==0)
        hat_y_t2=1;
    end
    z_1 = (hat_y_t1~=yt);
    z_2 = (hat_y_t2~=yt);
    u_t=u_t*beta^z_1;
    v_t=v_t*beta^z_2;
    id_new = id;
    x_t = X(id_new, :);
    y_t = Y(id_new);
    x_t = x_t*Pt;
    x_t = x_t*(1/sqrt(sum(x_t.^2,2)));
    
    l_t2 = max(0,1-yt*F_t);
    if (l_t2 > 0),
        s2_t = norm(x_t)^2;
        gamma_t = min(C,l_t2/s2_t);    
        Wt = Wt + gamma_t*yt*x_t;
    end

    id_new = id;
    x_t = X(id_new, :);
    y_t = Y(id_new);
    if temp ==1
        mean_Xt = (mean_Xt .* NUM_t + x_t)./(NUM_t + 1);
        NUM_t = NUM_t + 1;
        
        mean_kt(y_t,:) =(mean_kt(y_t,:).*NUM_kt(y_t) + x_t)./(NUM_kt(y_t) + 1);
        NUM_kt(y_t) = NUM_kt(y_t) + 1;
    end
    
    if temp == 0
        mean_Xt = x_t;
        mean_kt = zeros(options.k,14);
        NUM_kt = zeros(options.k,1);
        NUM_t = 1;
        temp = 1;
    end
    
    X_t =  MEAN_Xt0 - mean_Xt;
    X_kt = MEAN_kt0 - mean_kt;
    
    if mod(t,Tw)==0
        B = eye(size(x_t,2),size(x_t,2)) + mu*X_t'*X_t;
        for j=1:k
            B = B + mu*X_kt(j,:)'*X_kt(j,:);
        end
        if(det(B)~=0)
            Pt = inv(B)*Pt;
        else
            Pt = pinv(B)*Pt;
        end
    end   
end

y(y==1)=0;
y(y==2)=1;
y_p(y_p==-1)=0;

TP = sum(y == 1 & y_p == 1);
FP = sum(y == 0 & y_p == 1);
TN = sum(y == 0 & y_p == 0);
FN = sum(y == 1 & y_p == 0);

precision = TP / (TP + FP+0.0001);
recall = TP / (TP + FN+0.0001);
score = 2 * (precision * recall) / (precision + recall+0.0001);
sensitivity = TP / (TP + FN);
specificity = TN / (TN + FP);
G_mean = sqrt(sensitivity * specificity);
