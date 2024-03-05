function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs, ...
    accuracy,score,precision,recall,MCC,G_mean] = HomOTLMS(Y, X1, X2, options, id_list,classifiers)

%% initialize parameters
beta = options.beta2; 
Number_old=options.Number_old;
C = options.C; 
T_TICK = options.t_tick;

numSources = length(classifiers);

u_t = 1/(numSources + 1);
v_t = 1/(numSources + 1);
u_t_i = [];
for i = 1:length(classifiers)
    ws{i} = classifiers(i).w;
    u_t_i = [u_t_i, 1/(numSources + 1)];
end
for i = 1:numSources,
    p1_ts{i} = u_t_i(i) / (sum(u_t_i, 2) + v_t);
end
p2_t = v_t / (sum(u_t_i, 2) + v_t);

w2 = zeros(1, size(X2, 2));

numSV = 0;

SV2 = [];
ID = id_list;
err_count = 0;
y_p = [];
y = [];

mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
accu = [];
prec = [];
rec = [];
fm = [];
mcc = [];

t_tick = T_TICK; %10;
%% loop
tic
for t = 1:length(ID),
    id = ID(t);

    for i = 1:length(classifiers)
        x1_t = X1(id, :);
        f1_ts{i} = ws{i}*x1_t'; 
        
        p1_ts{i} = u_t_i(i) / (sum(u_t_i, 2) + v_t);
    end
    p2_t = v_t / (sum(u_t_i, 2) + v_t);
    
    id2=id-Number_old;
    x2_t = X2(id2, :);
    f2_t = w2*x2_t'; 
    
    f_t = 0;
    for i = 1:length(f1_ts)
        f_t = f_t + p1_ts{i}*sign(f1_ts{i});
    end
    f_t = f_t + p2_t * sign(f2_t);

    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    if (hat_y_t==1)
        hat_y_t=2;
    end
    if (hat_y_t==-1)
        hat_y_t=1;
    end
    % count accumulative mistakes
    if (hat_y_t~=Y(id)),
        err_count = err_count + 1;
    end

    y_p = [y_p hat_y_t];
    y = [y Y(id)];

    if (Y(id)==1)
        Y_t=-1;
    end
    if (Y(id)==2)
        Y_t=1;
    end
    
    z_v = (Y_t * f2_t < 0);
    v_t = v_t * beta^z_v;
    
    for i = 1:length(f1_ts)
        z_u_i = (Y_t * f1_ts{i} < 0);
        u_t_i(i) = u_t_i(i) * beta^z_u_i;
    end
    
    l2_t = max(0,1-Y_t*f2_t);   % hinge loss
    if (l2_t>0)
        % update      
        s2_t = norm(x2_t)^2;
        gamma_t = min(C,l2_t/s2_t);    
        w2 = w2 + gamma_t*Y_t*x2_t;
        numSV = numSV + 1;
    end
    run_time=toc;
    if t<T_TICK  
        if (t==t_tick)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
           
            TMs=[TMs run_time];
            
            t_tick=2*t_tick;
            if t_tick>=T_TICK,
                t_tick = T_TICK;
            end
            
        end
    else
        if (mod(t,t_tick)==0)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            TMs=[TMs run_time];
        end
    end
end

w_temp = [];
for i = 1:length(classifiers)
    w_temp = [w_temp ws{i}];
end

accuracy = 1 - err_count/t;
y(y==1)=0;
y(y==2)=1;
y_p(y_p==1)=0;
y_p(y_p==2)=1;

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

classifier.numSV = numSV;
classifier.w1 = w_temp;
classifier.w2 = w2;
run_time = toc;
