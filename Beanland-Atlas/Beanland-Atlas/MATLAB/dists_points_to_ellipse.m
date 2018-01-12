function [ dists ] = dists_points_to_ellipse(x, y, ra, rb, xc, yc, phi, accuracy)
%Find the distances of a set of points from an ellipse
%x is a list x ordinates; y is the list of corresponding yordinates
%xc is the ellipse centre x ordinate, yc is the y centre ordinate, ra is the semi-major axis length
%rb is the semi-minor axis length, phi is the orientation of the ellipse major axis with respect to
%the x axis (in radians).

%%Get the distances of each point from the ellipse
[h, ~] = size(xdata);
dists = zeros(h, 1);
for i = 1:h
    dists(i, 1) = distancePointToEllipse(x(i, 1), y(i, 1), ra, rb, xc, yc, phi, accuracy);
end
end