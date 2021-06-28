function x_temp = bisectionMethodNew(gradient_gamma, v, min_bis, max_bis, error_bisection, tt0, w, l, a, r)

max_iter = 1000;
x_temp = (min_bis+max_bis)/2;

gradient_temp = gradient_gamma(x_temp, v, tt0, w, l, a, r);
counter = 0;
% Check gradient at min_bis and max_bis
gradient_temp_min = gradient_gamma(min_bis, v, tt0, w, l, a, r);
gradient_temp_max = gradient_gamma(max_bis, v, tt0, w, l, a, r);
if gradient_temp_min>0
    x_temp = min_bis;
elseif gradient_temp_max<0
    x_temp = max_bis;
else
%     while abs(max_bis-min_bis)>error_bisection
    while abs(gradient_temp)>error_bisection
    %     if f_obj(c, s0, w, L, l, a, r)<0&&f_obj(min_bis, s0, w, L, l, a, r)<0
    %     if f_obj(c, s0, w, L, l, a, r)<0
        if gradient_temp < 0
            min_bis = x_temp;
        else
            max_bis = x_temp;
        end
        x_temp = (min_bis+max_bis)/2;
        gradient_temp = gradient_gamma(x_temp, v, tt0, w, l, a, r);
        counter = counter + 1;
        if counter > max_iter
            break
        elseif x_temp<1
            x_temp = 0;
            break
        end
    end
end
