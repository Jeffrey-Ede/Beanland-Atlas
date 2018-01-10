#include <commensuration_ellipses.h>

namespace ba
{
	/*Get ellipses describing each spot from their Scharr filtrates. Ellipses are checked using heuristic arguments:
	**ellipse shapes vary smoothly with time and ellipse shaps must be compatible with a projection of an array of
	**circles onto a flat detector
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to extract spots from
	**spot_pos: std::vector<cv::Point> &, Positions of located spots in aligned diffraction pattern
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 3 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each image, each spot in the order of their positions in the positions vector, set of
	**3 points desctribing the ellipse, in that order
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	*/
	void get_spot_ellipses(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, cv::Mat &acc, 
		std::vector<std::vector<std::vector<cv::Point>>> &ellipses)
	{
		//Use the Scharr filtrate of the aligned diffraction patterns to estimate the ellipses
		std::vector<std::vector<cv::Point>> acc_ellipses;
		//ellipse_sizes(acc, spot_pos, acc_ellipses);

		//Use the aligned image Scharr filtrate to get approximate information about the ellipses

		cv::Mat scharr;
		scharr_amp(acc, scharr);


	}

	/*Amplitude of image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Floating point image to get the Scharr filtrate of
	**scharr_amp: cv::Mat &, OpenCV mat to output the amplitude of the Scharr filtrate to
	*/
	void scharr_amp(cv::Mat &img, cv::Mat &scharr_amp)
	{
		//Aquire the image's Sobel filtrate
		cv::Mat gradx, grady;
		cv::Sobel(img, gradx, CV_32FC1, 0, 1, CV_SCHARR);
		cv::Sobel(img, grady, CV_32FC1, 1, 0, CV_SCHARR);

		//Sum the gradients in quadrature
		float *p, *q;
        #pragma omp parallel for
		for (int i = 0; i < img.rows; i++)
		{
			p = gradx.ptr<float>(i);
			q = grady.ptr<float>(i);
			for (int j = 0; j < img.cols; j++)
			{
				p[j] = std::sqrt(p[j]*p[j] + q[j]*q[j]);
			}
		}

		scharr_amp = gradx;
	}

	/*Use weighted sums of squared differences to calculate the sizes of the ellipses from an image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Image to find the size of ellipses at the estimated positions in
	**spot_pos: std::vector<cv::Point>, Positions of spots in the image
	**est_rad: std::vector<cv::Vec3f> &, Two radii to look for the ellipse between
	**est_frac: const float, Proportion of highest Scharr filtrate values to use when initially estimating the ellipse
	**ellipses: std::vector<ellipse> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 3 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps, if necessary. The nesting is each spot in the order of their positions in the positions vector, set of 3 points
	**(1 is extra) desctribing the ellipse, in that order
	**ellipse_thresh_frac: const float, Proportion of Schaar filtrate in the region use to get an initial estimage of the ellipse
	**to use
	*/
	void get_ellipses(cv::Mat &img, std::vector<cv::Point> spot_pos, std::vector<cv::Vec3f> est_rad, const float est_frac,
		std::vector<ellipse> &ellipses, const float ellipse_thresh_frac)
	{
		//Calculate the amplitude of the image's Scharr filtrate
		cv::Mat scharr;
		scharr_amp(img, scharr);

		ellipses = std::vector<ellipse>(spot_pos.size());

		//For each ellipse on the image 
		for (int i = 0; i < spot_pos.size(); i++)
		{
			//Extract the region where the ellipse is located
			cv::Mat mask, annulus;
			create_annular_mask(mask, 2*est_rad[i][1]+1, est_rad[i][0], est_rad[i][1]);
			get_mask_values( img, annulus, mask, cv::Point2i( spot_pos[i].y-est_rad[i][1], spot_pos[i].y-est_rad[i][1] ) );

			//Use weighted hyper-renormalisation to fit a conic to the data
			std::vector<double> conic = hyper_renorm_conic(mask, annulus, 0.5*(est_rad[i][0]+est_rad[i][1]));

			//Store the ellipse parameters
			ellipses[i] = ellipse_points_from_conic(conic);
		}
	}

	/*Create annular mask
	**Inputs:
	**size: const int, Size of the mask. This should be an odd integer
	**inner_rad: const int, Inner radius of the annulus
	**outer_rad: const int, Outer radius of the annulus
	**val, const byte, Value to set the elements withing the annulus. Defaults to 1
	*/
	void create_annular_mask(cv::Mat annulus, const int size, const int inner_rad, const int outer_rad, const byte val)
	{
		//Create mask
		annulus = cv::Mat::zeros(cv::Size(size, size), CV_8UC1);

		//Position of mask center
		cv::Point origin = cv::Point((int)(size/2), (int)(size/2));

		//Set the elements in the annulus to the default value
		byte *p;
        #pragma omp parallel for
		for (int i = 0; i < size; i++)
		{
			p = annulus.ptr<byte>(i);
			for (int j = 0; j < size; j++)
			{
				//Mark the position if it is in the annulus
				float dist = std::sqrt((i-origin.y)*(i-origin.y) + (j-origin.x)*(j-origin.x));
				if (dist >= inner_rad && dist <= outer_rad)
				{
					p[j] = val;
				}
			}
		}
	}

	/*Extracts values at non-zero masked elements in an image, constraining the boundaries of a mask so that only maked 
	**points that lie in the image are extracted. It is assumed that at least some of the mask is on the image
	**Inputs:
	**img: cv::Mat &, Image to apply the mask to
	**dst: cv::Mat &, Extracted pixels of image that mask has been applied to
	**mask: cv::Mat &, Mask being applied to the image
	**top_left: cv::Point2i &, Indices of the top left corner of the mask on the image
	*/
	void get_mask_values(cv::Mat &img, cv::Mat &dst, cv::Mat &mask, cv::Point2i &top_left)
	{
		//Get the limits of the mask to iterate between so that it doesn't go over the sides of the image
		int llimx = top_left.x >= 0 ? 0 : -top_left.x;
		int ulimx = top_left.x+mask.cols <= img.cols ? mask.cols : img.cols-top_left.x;
		int llimy = top_left.y >= 0 ? 0 : -top_left.y;
		int ulimy = top_left.y+mask.cols <= img.cols ? mask.cols : img.cols-top_left.y;
		
		//Constrain the top left position accordingly
		int top_leftx = llimx ? 0: top_left.x;
		int top_lefty = llimy ? 0: top_left.y;

		dst = img(cv::Rect(top_leftx, top_lefty, ulimx-llimx, ulimy-llimy))(mask(cv::Rect(0, 0, ulimx-llimx, ulimy-llimy)));
	}

	/*Threshold a proportion of values using an image histogram
	**Inputs:
	**img: cv::Mat &, Image of floats to threshold
	**thresh: cv::Mat &, Output binary thresholded image
	**thresh_frac: const float, Proportion of the image to threshold
	**thresh_mode: const int, Type of binarialisation. Defaults to cv::THRESH_BINARY_INV, which marks the highest 
	**proportion
	**hist_bins: const int, Number of bins in histogram used to determine threshold
	**non-zero: const bool, If true, only use non-zero values when deciding the threshold. Defaults to false.
	**Returns:
	**unsigned int, Number of points in the thresholded image
	*/
	unsigned int threshold_proportion(cv::Mat &img, cv::Mat &thresh, const float thresh_frac, const int thresh_mode,
		const int hist_bins, const bool non_zero)
	{
		//Get the range of values the threshold is being applied to
		float non_zero_compensator;
		unsigned int num_non_zero = 0;
		double min, max;
		if (non_zero)
		{
			//Calculate the minimum non-zero value and the maximum value, recording the number of non-zero values
			float *p;
			float min = FLT_MAX;
			for (int i = 0; i < img.rows; i++)
			{
				p = img.ptr<float>(i);
				for (int j = 0; j < img.cols; j++)
				{
					//Check if the element is non-zero
					if (p[j])
					{
						num_non_zero++;

						//Check if it is the minimum non-zero value
						if (p[j] < min)
						{
							min = p[j];
						}
					}
				}
			}

			non_zero_compensator = img.rows * img.cols / num_non_zero;
		}
		else
		{
			num_non_zero = img.rows * img.cols; //It doesn't matter if elements are zero or not
			non_zero_compensator = 1.0f; //It doesn't matter if elements are zero or not

			cv::minMaxLoc(img, &min, &max, NULL, NULL);
		}

		//Calculate the image histogram
		cv::Mat hist;
		int hist_size = non_zero_compensator * hist_bins;
		float range[] = { min, max };
		const float *ranges[] = { range };
		cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &hist_size, ranges, true, false);

		//Work from the top of the histogram to calculate the threshold to use
		float thresh_val;
		const int use_num = thresh_frac * num_non_zero;
		for (int i = hist_size-1, tot = 0; i >= 0; i--)
		{
			//Accumulate the histogram bins
			tot += hist.at<float>(i, 1);

			//If the desired total is exceeded, record the threshold
			if (tot > use_num)
			{
				thresh_val = i * (max - min) / hist_size;
				break;
			}
		}

		//Threshold the estimated symmetry center values
		cv::threshold(img, thresh, thresh_val, 1, thresh_mode);

		return num_non_zero;
	}

	/*Weight the fit an ellipse to a noisy set of data using hyper-renormalisation. The function generates the coefficients
	**A0 to A5 when the data is fit to the equation A0*x*x + 2*A1*x*y + A2*y*y + 2*f0*(A3*x + A4*y) + f0*f0*A5 = 0, in
	**that order
	**Inputs:
	**mask: cv::Mat &, 8-bit mask. Data points to be used are non-zeros
	**weights: cv::Mat &, 32-bit mask. Weights of the individual data points
	**f0: const double, Approximate size of the ellipse. This is arbitrary, but choosing a value close
	**to the correct size reduces numerical errors
	**thresh: const double, Iterations will be concluded if the cosine of the angle between successive eigenvectors divided
	**by the amplitude ratio (larger divided by the smaller) is smaller than the threshold
	**max_iter: const int, The maximum number of iterations to perform. If this limit is reached, the last iteration's conic
	**coefficients will be returned
	**Returns:
	**std::vector<double>, Coefficients of the conic equation
	*/
	std::vector<double> hyper_renorm_conic(cv::Mat &mask, cv::Mat weights, const double f0, const double thresh, 
		const int max_iter)
	{
		//Initialise the MATLAB engine
		matlab::data::ArrayFactory factory;
		std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::connectMATLAB();

		//Package the mask and weights into vectors so that they can be passed to MATLAB
		std::vector<byte> mask_vect(mask.rows*mask.cols);
		std::vector<double> weights_vect(mask.rows*mask.cols);
		byte *p;
		float *q;
		for (int i = 0, k = 0; i < mask.rows; i++)
		{
			p = mask.ptr<byte>(i);
			q = weights.ptr<float>(i);
			for (int j = 0; j < mask.cols; j++, k++)
			{
				mask_vect[k] = p[j];
				weights_vect[k] = (double)q[j];
			}
		}

		//Package data for the cubic Bezier profile calculator
		std::vector<matlab::data::Array> args({
			factory.createArray( { (size_t)mask.rows, (size_t)mask.cols }, mask_vect.begin(), mask_vect.end() ), //Points to fit
			factory.createArray( { (size_t)mask.rows, (size_t)mask.cols }, weights_vect.begin(), weights_vect.end() ), //Point weights
			factory.createScalar<double>(f0), //Size scale
			factory.createScalar<double>(thresh), //Minimum desired similarity of parameters from successive iterations
			factory.createScalar<int32_t>(max_iter) //Maximum number of iterations
		});

		//Pass data to MATLAB to calculate the cubic Bezier profile
		matlab::data::TypedArray<double> const conic = matlabPtr->feval(
			matlab::engine::convertUTF8StringToUTF16String("hyper_renorm_conic"), args);

		//Return the conic coefficients in an easy-to-use vector
		std::vector<double> conic_coeff(6);
		{
			int k = 0;
			for (auto val : conic)
			{
				conic_coeff[k] = val;
				k++;
			}
		}

		return conic_coeff;
	}

	/*Calculate the center and 4 extremal points of an ellipse (at maximum and minimum distances from the center) from
	**the coefficients of the conic equation A*x*x + B*x*y + Cy*y + D*x + E*y + F = 0
	**Input:
	**conic: std::vector<double> &, Coefficients of the conic equation
	**Returns:
	**Ellipse, Points describing the ellipse. If the conic does not describe an ellipse, the ellipse is
	**returned empty
	*/
	ellipse ellipse_points_from_conic(std::vector<double> &conic)
	{
		//Check that the conic equation describes an ellipse
		ellipse el;
		if (4.0*conic[0] * conic[2] - conic[1] * conic[1] > 0)
		{
			el.is_ellipse = false;
			return el;
		}
		else
		{
			el.is_ellipse = true;

			//Get 2 times the ellipse's angle of rotatation
			double theta_times_2 = std::atan( conic[1] / ( conic[0]-conic[2] ) );

			//Get the rotation angle in the first quadrant
			if (theta_times_2 < 0)
			{
				theta_times_2 += 0.5*PI;
			}

			//Construct the ellipse

			//Record the angle of rotation
			el.angle = 0.5*theta_times_2;

			//Get the sine and cosine of theta 
			double cos_theta = std::cos(el.angle);
			double sin_theta = std::sin(el.angle);

			//Create an alternative set of coefficients where the rectangular term B = 0
			double A, C, D, E, F;

			A = conic[0]*cos_theta*cos_theta + conic[1]*cos_theta*sin_theta + conic[2]*sin_theta*sin_theta;
			C = conic[0]*sin_theta*sin_theta - conic[1]*cos_theta*sin_theta + conic[2]*cos_theta*cos_theta;
			D = conic[3]*cos_theta + conic[4]*sin_theta;
			E = conic[4]*cos_theta - conic[3]*sin_theta;
			F = conic[5];

			//Get the ellipse center for these alternative coefficients
			double x = -0.5 * D / A;
			double y = -0.5 * E / C;

			//Record the actual center of the ellipse
			el.center = cv::Point2d( x*cos_theta - y*sin_theta, x*sin_theta + y*cos_theta );

			//Get the elongation factors for the alternative coefficients (a = b when there is no rectangular term)
			double num = -4.0*F*A*C + C*D*D + A*E*E;
			el.a = std::sqrt( num / ( 4.0*A*C*C ) );
			el.b = std::sqrt( num / ( 4.0*A*A*C ) );

			//Rotate anticlockwisely back to the original frame to get the positions of the points from the bottom
			//left, going clockwise
			std::vector<cv::Point2d> extrema(4);
			extrema[1] = el.center + cv::Point2d( -el.b*sin_theta, el.b*cos_theta );
			extrema[2] = el.center + cv::Point2d( el.a*cos_theta, el.a*sin_theta );
			extrema[3] = el.center + cv::Point2d( el.b*sin_theta, -el.b*cos_theta );
			extrema[4] = el.center + cv::Point2d( -el.a*cos_theta, -el.a*sin_theta );

			return el;
		}
	}

	/*Rotate a point anticlockwise
	**Inputs:
	**point: cv::Point &, Point to rotate
	**angle: const double, Angle to rotate the point anticlockwise
	**Returns:
	**cv::Point2d, Rotated point
	*/
	cv::Point2d rotate_point2D(cv::Point2d &point, const double angle)
	{
		return cv::Point2d(point.x * std::cos(angle) - point.y * std::sin(angle), 
			point.x * std::sin(angle) + point.y * std::cos(angle));
	}

	/*Exploit the inverse square law to find the sign of the inciding angle from an image's bacgkround
	**Inputs:
	**img: cv::Mat, Diffraction pattern to find which side the electron beam is inciding from
	**img_spot_pos: std::vector<cv::Point2d> &, Approximate positions of the spots on the image
	**fear: const float, Only use background at least this distance from the spots
	**dir: cv::Vec2d &, Vector indicating direction of maximum elongation due to the incidence angularity
	**Returns:
	**double, +/- 1.0: +1 means that elongation is in the same direction as decreasing intensity
	*/
	double inv_sqr_inciding_sign(cv::Mat img, std::vector<ellipse> &ellipses, const float fear, 
		cv::Vec2d &dir)
	{
		//Calculate angle between the elongation vector and the horizontal
		double angle = std::asin( dir[0] / std::abs(dir[0]*dir[0] + dir[1]*dir[1]) );

		//Rotate the image so that sums are less complicated
		cv::Mat rot = rotate_CV (img, dir[1] > 0 ? angle : 2*PI-angle );

		//Draw black circles over the spots
		for (int i = 0; i < ellipses.size(); i++)
		{
			double rad = std::max(ellipses[i].a, ellipses[i].b) + fear;
			cv::Point2d loc = rotate_point2D(ellipses[i].center-cv::Point2d(img.cols/2, img.rows/2), -angle) + 
				cv::Point2d((rot.cols-img.cols)/2, (rot.rows-img.rows)/2);
			cv::circle(img, loc, rad, cv::Scalar(0.0), -1, 8, 0);
		}

		//Sum along the rows of the rotated matrix, skipping zero-valued elements
		float *p;
		std::vector<float> row_sums;
        #pragma omp parallel for
		for (int i = 0; i < rot.rows; i++)
		{
			int num_contrib = 0;
			p = rot.ptr<float>(i);
			for (int j = 0; j < rot.cols; j++)
			{
				//If the element is non-zero
				if (p[j])
				{
					row_sums[i] += p[j];
					num_contrib++;
				}
			}

			row_sums[i] /= num_contrib;
		}

		/* Set up matrices to find the least squares solution that fits the intensity profile */

		Eigen::VectorXd x(row_sums.size());
		for (int i = 0; i < row_sums.size(); i++)
		{
			x(i) = i;
		}

		Eigen::MatrixXd X(row_sums.size(), 2);
		Eigen::VectorXd Xtemp(2);
		for (int i = 0; i < row_sums.size(); i++)
		{
			Xtemp << std::pow(x(i),2), 1;
			X.row(i) = Xtemp;
		}

		Eigen::VectorXd D(row_sums.size());
		for (int i = 0; i < row_sums.size(); i++)
		{
			D(i) = row_sums[i];
		}

		//Find the least squares solutions
		Eigen::VectorXd a = X.colPivHouseholderQr().solve(D);

		//Direction is the same as that of decreasing intensity
		if (a(1) < 0)
		{
			return 1.0;
		}
		//Direction is opposite that of decreasing intensity
		else
		{
			return -1.0;
		}
	}
}