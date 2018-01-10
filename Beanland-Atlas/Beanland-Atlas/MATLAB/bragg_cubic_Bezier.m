function [ profile ] = bragg_cubic_Bezier( xdata1, xdata2, ydata, r, tol, max_iter)
%Fit a symmetric, monotonically decreasing cubic Bezier curve to an intensity
%profile overlapping with itself

%Cast to double precision for lsqcurvefit()
xdata1 = double(xdata1);
xdata2 = double(xdata2);
ydata = double(ydata);
r_d = double(r);

%Ratio of 2 Bezier curves
fun = @(param,xdata)get_ratio(param, [xdata1 xdata2], r_d);

%Initial estimate of the fitting parameters
x0 = [0.5*r_d, 0.5, 0.5, 0.5, 0.5];

%Lower and upper bounds
lb = [0.0, 0.0, 0.0, 0.0, 0.0];
ub = [r_d, 1.0, 1.0, 1.0, 1.0];

%Fit the data
options = optimoptions(@lsqcurvefit, 'FunctionTolerance', tol, 'MaxIterations', ...
    max_iter, 'Display', 'off');
param = lsqcurvefit(fun, x0, [xdata1 xdata2], ydata, lb, ub, options);

%Use the parameters to calculate the circular, angle-independent condenser
%lens profile
profile = zeros(2*r+1, 2*r+1);
for i = 0:r
    for j = 0:min(floor(sqrt(double(r*r-i*i))), double(i))
        %Distance from the circle center
        dist = sqrt(double(i*i+j*j));

        %Get the profile value
        y = get_y(dist, r_d, param(1), param(2), param(3), param(4), param(5));

        %Set the symmetrically equivalent element values
        profile(r+i+1, r+j+1) = y;
        profile(r+i+1, r-j+1) = y;
        profile(r-i+1, r+j+1) = y;
        profile(r-i+1, r-j+1) = y;

        profile(r+j+1, r+i+1) = y;
        profile(r+j+1, r-i+1) = y;
        profile(r-j+1, r+i+1) = y;
        profile(r-j+1, r-i+1) = y;
    end
end
end

function [ t ] = get_t(x, x1, x2, r)
%Get the roots, t, corresponding to x and the choices of coefficients
cubic_roots = roots([(r+3*x1-3*x2) (3*x2-6*x1) (3*x1) (-x)]);

%Get the real roots
cubic_roots = cubic_roots(imag(cubic_roots) == 0);

%The solution is the non-imaginary root within the relevant x range
t = 1E10; %A large number
for i = 1:numel(cubic_roots)
   if 0 <= cubic_roots(i) && cubic_roots(i) <= 1.00
       t = cubic_roots(i);
       break;
   end
end

%Ensure that t is assigned, preventing problems caused by rounding errors
if t == 1E10
    d = cubic_roots(1) - 0.5;
    if numel(cubic_roots) > 1
        for i = 2:numel(cubic_roots)
           if abs(cubic_roots(i) - 0.5) < abs(d)
               d = cubic_roots(i) - 0.5;
           end
        end
    end
    %Assign t the value of the limit it passed due to rounding errors
    if d >= 0.5
        t = 1.0;
    else
        t = 0.0;
    end
end
end

function [ y ] = get_y(x, r, x1, a2, b1, b2, b3)
%Get the x ordinate corresponding to a2
x2 = x1 + a2*(r-x1);

%Get the parametric position on the Bezier curve for these x
t = get_t(x, x1, x2, r);

%Get the gradient of the lower bounding line
m = (b3 - 1.0) / r;

%y ordninates of the unknown Bezier control points
y1 = (1-b1)*m*x1 + 1.0;
y2 = b2*y1 + (1-b2)*(m*x2 + 1.0);

y = (1-t)^3 + 3*(1-t)^2*t*y1 + 3*(1-t)*t^2*y2 + t^3*b3;
end

function [ ratio ] = get_ratio(param, xdata, r)
%Get the ratios of y value at the first x position to the second
[h, ~] = size(xdata);
ratio = zeros(h, 1);
for i = 1:h
    ratio(i, 1) = get_y(xdata(i, 1), r, param(1), param(2), param(3), param(4), param(5)) / ...
       get_y(xdata(i, 2), r, param(1), param(2), param(3), param(4), param(5));
end
end