function [classifier] = avePA1_K_M(Y, X, options, id_list)

%% initialize parameters
C = options.C;
w = zeros(1,size(X,2));
ave_w = zeros(1,size(X,2));
ID = id_list;
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    x_t = X(id,:);
    y_t = Y(id);
    F_t = w*x_t';

    if (Y(id)==1)
        yt = -1;
    end
    if (Y(id)==2)
        yt = 1;
    end
    l_t = max(0,1-yt*F_t);   
    hat_y_t = sign(F_t); 
    if (hat_y_t==0)
        hat_y_t=1;
    end
    if (l_t > 0),
        s_t = norm(x_t)^2;
        gamma_t = min(C,l_t/s_t);
        
        w = w + gamma_t*yt*x_t;
        
        ave_w = ((t-1)/t)*ave_w + (1/t)*w;
    else
        ave_w = ((t-1)/t)*ave_w + (1/t)*w;
    end
end
classifier.w = ave_w;
