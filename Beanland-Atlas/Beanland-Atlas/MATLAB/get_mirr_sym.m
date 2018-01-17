function lines = get_mirr_sym(ratios, mask, line0, lower, upper, tol, max_iter)
%Get the mirror symmetry lines for a monotonically decreasing Bezier surface 
%overlapping with its translation

%Cast to double precision for lsqcurvefit()
xdata = double(ratios);

%Adjust from C++ to MATLAB indexing
line0 = double(line0 + 1.0);

%Ratios of values on either side of a mirror line
fun = @(param)get_line_ratios(param, xdata, mask, line0);

%%Estimate mirror line parameters
center_x = 0.5*(line0(1) + line0(3));
center_y = 0.5*(line0(2) + line0(4));

%Lines are expected to be perpendicular; however, this will not be imposed
x0_1 = [ 0.0, center_x, center_y ];
x0_2 = [ 0.5*pi, center_x, center_y ];

%Lower and upper bounds
lb = double([lower(1), lower(2), lower(3)]);
ub = double([upper(1), upper(2), upper(3)]);

%The parameters being varied have been chosen so that there are no linear constraints
A = [];
b = [];
Aeq = [];
beq = [];

%Fit the data
options = optimoptions(@lsqcurvefit, 'FunctionTolerance', tol, 'MaxIterations', ...
    max_iter, 'Display', 'off');
line1 = fmincon(fun, x0_1, A, b, Aeq, beq, lb(1:2), ub(1:2), options);
line2 = fmincon(fun, x0_2, A, b, Aeq, beq, lb(3:4), ub(3:4), options);

%Adjust from MATLAB to C++ indexing
lines = [ (line1-1) (line2-1) ];
end

function [ var ] = get_line_ratios(param, xdata, mask, line0)
%Get the variance of the ratios across the mirror line from unity

%Calculate the positions of points of the transformed line
line = shift_and_rot_line(param, line0);

%%Combinations of the line parameters
%Get differences between points the line goes thought
x2_m_x1 = line(3)-line(1);
y2_m_y1 = line(4)-line(2);

%%Get ratios of pixels to those reflected by the mirror line
var = 0.0;
count = 0.0;
[h, w] = size(xdata);
Y = [1:h];
X = [1:w];
ratio = 0.0;
for i = 1:w
    for j = 1:h
        %If there is intensity ratio information about the point
        if mask(j, i)
            %If the point is on one side of the line, not the other
            if (i-line(1))*y2_m_y1 - (j-line(2))*x2_m_x1 > 0
                %Calculate the position of the pixel in the mirror
                m = mirror(i, j, line(0), line(1), line(2), line(3));
                
                %If the points are at least 1.0 px apart (to avoid comparing 
                %pixels with themselves)
                if sqrt((m(1)-i)*(m(1)-i) + (m(2)-j)*(m(2)-j)) > 1.0
                    
                    %Check that the pixel in the mirror is in the image
                    if m(1) >= 1 && m(1) <= w && m(2) >= 1 && m(2) <= h
                        
                        %Check that the pixels needed for area
                        %interpolation are marked on the mask
                        if ~mask(floor(m(2)), floor(m(1)))
                           continue;
                        end
                        if floor(m(1)) ~= m(1)
                            if ~mask(floor(m(2)), floor(m(1))+1)
                                continue;
                            end
                        end
                        if floor(m(2)) ~= m(2)
                            if ~mask(floor(m(2))+1, floor(m(1)))
                                continue;
                            end
                        end
                        if floor(m(1)) ~= m(1) && floor(m(2)) ~= m(2)
                            if ~mask(floor(m(2))+1, floor(m(1))+1)
                                continue;
                            end
                        end
                        
                        %Interpolate the value at the point
                        val = interp2(Y, X, xdata, m(2), m(1), 'spline');
                        
                        %Take the ratio of the higher value to the smaller
                        %so the measurement is independent of the side of
                        %the line being mirrored from
                        if val > xdata(j, i)
                            ratio = val /  xdata(j, i);
                        else
                            ratio = xdata(j, i) / val;
                        end
                        
                        %Update the sum of squared differences from unity
                        %and the contribution count
                        var = var + (ratio-1.0)*(ratio-1.0);
                        count = count + 1;
                    end
                end
            end
        end
    end
end

%Divide the sum of squared differences by the number of degrees of freedom.
%It is count; not count-1, as the data is being matched against an
%expectation of unity
var = var / count;
end

function point = mirror(x, y, x0, y0, x1, y1)
%Reflect a point, [x y], in a line passing through the points
%[x0, y0] and [x1, y1]

dx = x1 - x0;
dy = y1 - y0;

a = (dx * dx - dy * dy) / (dx * dx + dy*dy);
b = 2.0 * dx * dy / (dx*dx + dy*dy);

x2 = a * (x - x0) + b*(y - y0) + x0;
y2 = b * (x - x0) - a*(y - y0) + y0;

point = [x2, y2]; 
end

function line = shift_and_rot_line(param, line)
%Shift a line and rotate it about its center
%param - [ shift x, shift y, angle to rotate anticlockwise in rad ]
%line - [ x1, y1, x2, y2 ]

%Get the center of the 2 points
x0 = 0.5*(line(1)+line(3));
y0 = 0.5*(line(2)+line(4));

%Subtract the center position from the line points so that the positions
%are relative to the origin, making rotation easy
x1 = line(1) - x0;
y1 = line(2) - y0;
x2 = line(3) - x0;
y2 = line(4) - y0;

%Rotate the points about the origin by the specified angle
x1 = x1*cos(param(3))-x1*sin(param(3));
y1 = y1*sin(param(3))+y1*cos(param(3));
x2 = x2*cos(param(3))-x2*sin(param(3));
y2 = y2*sin(param(3))+y2*cos(param(3));

%Translate the rotate points back to their original center and apply the
%shift
x1 = x1 + x0 + param(1);
y1 = y1 + y0 + param(2);
x2 = x2 + x0 + param(1);
y2 = y2 + y0 + param(2);

line = [ x1 y1 x2 y2 ];
end