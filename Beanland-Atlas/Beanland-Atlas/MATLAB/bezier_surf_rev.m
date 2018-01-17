function [ profile ] = bezier_surf_rev( xdata1, xdata2, ydata, r, tol, max_iter )
%Use the parameters to calculate the circular, angle-independent dynamical
%diffraction effect decoupled envelope

%Get the surface parameters
r_d = double(r);
param = bragg_cubic_Bezier( xdata1, xdata2, ydata, r_d, tol, max_iter );

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