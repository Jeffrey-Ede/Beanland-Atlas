#include <commensuration_tilts.h>

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
		ellipse_sizes(acc, spot_pos, acc_ellipses);

		//Use the 

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
	**spot_pos: std::vector<cv::Point>, Positions of located spots in the image
	**est_rad: std::vector<cv::Vec3f> &, Two radii to look for the ellipse between
	**est_frac: const float, Proportion of highest Scharr filtrate values to use when initially estimating the ellipse
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 3 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each spot in the order of their positions in the positions vector, set of 3 points
	**(1 is extra) desctribing the ellipse, in that order
	**ellipse_thresh_frac: const float, Proportion of Schaar filtrate in the region use to get an initial estimage of the ellipse
	**to use
	*/
	void ellipse_sizes(cv::Mat &img, std::vector<cv::Point> spot_pos, std::vector<cv::Vec3f> est_rad, const float est_frac,
		std::vector<std::vector<cv::Point>> &ellipses, const float ellipse_thresh_frac)
	{
		//Calculate the amplitude of the image's Scharr filtrate
		cv::Mat scharr;
		scharr_amp(img, scharr);

		ellipses = std::vector<std::vector<cv::Point>>(spot_pos.size());

		//For each ellipse on the image 
		for (int i = 0; i < spot_pos.size(); i++)
		{
			//Extract the region where the ellipse is located
			cv::Mat mask, annulus;
			create_annular_mask(mask, 2*est_rad[i][1]+1, est_rad[i][0], est_rad[i][1]);
			get_mask_values( img, annulus, mask, cv::Point2i( spot_pos[i].y-est_rad[i][1], spot_pos[i].y-est_rad[i][1] ) );

			hyper_renorm_ellipse(annulus, mask, annulus);

			////Threshold the 50% highest Scharr filtrate values to get an initial estimate for the ellipse
			//cv::Mat thresh;
			//unsigned int num_points = threshold_proportion(annulus, thresh, ellipse_thresh_frac, cv::THRESH_BINARY, 100, true);
			//
			////Get the points making up the ellipse
			//byte *p;
			//std::vector<cv::Point> points(num_points);
			//for (int j = 0, n = 0; j < thresh.rows; j++)
			//{
			//	p = thresh.ptr<byte>(j);
			//	for (int k = 0; k < thresh.cols; k++)
			//	{
			//		//If this is one of the thresholded points, add it to the vector
			//		if (p[k])
			//		{
			//			points[n] = cv::Point(k, j);
			//			n++;
			//		}
			//	}
			//}

			////Use the thresholded points to estimate the ellipse
			
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
	**mask: cv::Mat &, Data points to be used are non-zeros
	**weights: cv::Mat &, Weights of the individual data points
	**f0: const double, Approximate size of the ellipse. This is arbitrary, but choosing a value close
	**to the correct size reduces numerical errors
	**thresh: const float, Iterations will be concluded if the cosine of the angle between successive eigenvectors divided
	**by the amplitude ratio (larger divided by the smaller) is larger than the threshold
	**max_iter: const int, The maximum number of iterations to perform
	**Returns:
	**std::vector<double>, Coefficients of the conic equation
	*/
	std::vector<double> hyper_renorm_conic(cv::Mat &mask, cv::Mat weights, const double f0, const float thresh, 
		const int max_iter)
	{
		//Matrix symmetriciser
		auto sym = [](Eigen::MatrixXd matrix){ return (matrix + matrix.transpose()) / 2; };

		//Amplitude of the dot product between 2 eigenvectors
		auto dot_amp = [](Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType &e1, 
			Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType &e2)
		{ 
			std::complex<double> dot = e1(0, 0) * e2(0, 0);
			for (int i = 1; i < e1.rows(); i++)
			{
				dot += e1(i, 0) * e2(i, 0);
			}

			return std::sqrt(dot.real*dot.real + dot.imag*dot.imag);
		};

		Eigen::VectorXd e(6);
		e(0) = 1; e(1) = 0; e(2) = 1; e(3) = 0; e(4) = 0; e(5) = 0;

		//Set values of the V0 matrix elements that don't change
		Eigen::MatrixXd V0 = Eigen::MatrixXd::Constant(6, 6, 0.0);
		V0(3, 3) = 4*f0*f0;
		V0(4, 4) = 4*f0*f0;

		Eigen::MatrixXd M = Eigen::MatrixXd::Constant(6, 6, 0.0);
		Eigen::MatrixXd N = Eigen::MatrixXd::Constant(6, 6, 0.0);
		
		Eigen::VectorXd epsilon(6); 
		epsilon(5) = f0*f0; //This value is the same in every element of the sum
		
		//Construct the matrix M
		unsigned int num_points;
		byte *p;
		float *q;
		double sum_weights = 0.0;
		std::vector<Eigen::VectorXd> epsilons;
        #pragma omp parallel for reduction(sum:num_points), reduction(sum:sum_weights), reduction(sum:M)
		for (int y = 0, i = 0; y < mask.rows; y++)
		{
			p = mask.ptr<byte>(y);
			q = weights.ptr<float>(y);
			for (int x = 0; x < mask.cols; x++, num_points++, i++)
			{
				if (p[x])
				{
					//Calculate contribution to matrix M
					epsilon(0) = x*x;
					epsilon(1) = 2*y*x;
					epsilon(2) = y*y;
					epsilon(3) = 2*f0*x;
					epsilon(4) = 2*f0*y;

					epsilons[i] = epsilon;

					M += epsilon*epsilon.transpose();


					//Sum weights for matrix N calculation
					sum_weights += q[x];
				}
			}
		}

		//Get the eigenvalues and eigenvectors of M
		Eigen::EigenSolver<Eigen::MatrixXd> eigen_sol(M);

		//Structure to combine the eigenvalus and eigenvectors so that they can be coupled for sorting
		struct eigen_val_vect {
			double val;
			Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType vect;
		};

		//Custom comparison of coupled elements based on angle
		struct by_eigenvalue { 
			bool operator()(eigen_val_vect const &a, eigen_val_vect const &b) { 
				return a.val > b.val;
			}
		};

		//Populate the coupled structure
		std::vector<eigen_val_vect> eigen_val_vect_pairs(6);
		for (int i = 0; i < 6; i++)
		{
			eigen_val_vect_pairs[i].val = eigen_sol.eigenvalues()[i].real();
			eigen_val_vect_pairs[i].vect = eigen_sol.eigenvectors.col(i);
		}

		//Use the custom comparison to sort the coupled elements in order of decreasing eigenvalue
		std::sort(eigen_val_vect_pairs.begin(), eigen_val_vect_pairs.end(), by_eigenvalue());

		//Calculate the psuedoinverse of M of truncated rank 5
		Eigen::MatrixXd M5 = Eigen::MatrixXd::Constant(6, 6, 0.0);
		for (int i = 0; i < 5; i++)
		{
			M5 += ( eigen_val_vect_pairs[i].vect * eigen_val_vect_pairs[i].vect.transpose() ) / 
				eigen_val_vect_pairs[i].val;
		}

		//Normalise the weights so that they add up to 1
		cv::Mat norm_weights = weights / sum_weights;

		//Construct the matrix N
		std::vector<Eigen::MatrixXd> V(num_points, V0);
        #pragma omp parallel for reduction(sum:N)
		for (int y = 0, i = 0; y < mask.rows; y++)
		{
			p = mask.ptr<byte>(y);
			q = norm_weights.ptr<float>(y);
			for (int x = 0; x < mask.cols; x++, i++)
			{
				if (p[x])
				{
					//Store the V0 matrix so that it does not have to keep being recalculated
					V[i](0, 0) = 4*x*x;
					V[i](0, 1) = 4*x*y;
					V[i](0, 3) = 4*f0*x;
					V[i](1, 0) = 4*x*y;
					V[i](1, 1) = 4*(x*x + y*y);
					V[i](1, 2) = 4*x*y;
					V[i](1, 3) = 4*f0*y;
					V[i](1, 4) = 4*f0*x;
					V[i](2, 1) = 4*x*y;
					V[i](2, 2) = 4*y*y;
					V[i](2, 4) = 4*f0*y;
					V[i](3, 0) = 4*f0*x;
					V[i](3, 1) = 4*f0*y;
					V[i](4, 1) = 4*f0*x;
					V[i](4, 2) = 4*f0*y;

					//Add contribution to matrix N
					N += q[x] * ( V[i] + 2*sym(epsilon*e.transpose()) - (1 / (double)num_points) *
						( epsilon.dot(M5*epsilon) * V[i] + 2.0 * sym( V[i] * M5 * epsilon * epsilon.transpose() ) ) );
				}
			}
		}

		//Solve N * theta = 1/lambda * M * theta to get the eigenvector theta corresponding to the largest eigenvalue 1/lambda
		Eigen::EigenSolver<Eigen::MatrixXd> solution(M.inverse() * N);

		//Find the eigenvector with the highest eigenvalue
		double max_eigenvalue = solution.eigenvalues()[0].real;
		int max_eigenvalue_idx = 0;
		for (int i = 1; i < 6; i++)
		{
			if (solution.eigenvalues()[i].real > max_eigenvalue)
			{
				max_eigenvalue = solution.eigenvalues()[i].real;
				max_eigenvalue_idx = i;
			}
		}

		//Largest eigenvector
		Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType theta0 = solution.eigenvectors.col(max_eigenvalue_idx);

		//Repeat the calculation, iterativery converging from the HyperLS solution
		for (int k = 0; k < max_iter; k++)
		{
			//Prepare M and N matrices for recalculation
			M = Eigen::MatrixXd::Constant(6, 6, 0.0);
			N = Eigen::MatrixXd::Constant(6, 6, 0.0);

			//Recalculate the M and N matrices
            #pragma omp parallel for reduction(sum:N)
			for (int y = 0, i = 0; y < mask.rows; y++)
			{
				p = mask.ptr<byte>(y);
				q = norm_weights.ptr<float>(y);
				for (int x = 0; x < mask.cols; x++, i++)
				{
					if (p[x])
					{
						//Calculate the reweighting factor
						double weight = 1.0 / ( theta0.dot( V[i]*theta0 ).real() );

						M += weight * epsilons[i] * epsilons[i].transpose();

						//Add contribution to matrix N
						N += q[x] * ( weight * ( V[i] + 2*sym(epsilons[i]*e.transpose()) ) - (weight*weight / (double)num_points) *
							( epsilons[i].dot( M5*epsilons[i] ) * V[i] + 2.0 * sym( V[i] * M5 * epsilons[i] * epsilons[i].transpose() ) ) );
					}
				}
			}
			//Find the eigenvector with the highest eigenvalue
			max_eigenvalue = solution.eigenvalues()[0].real;
			max_eigenvalue_idx = 0;
			for (int i = 1; i < 6; i++)
			{
				if (solution.eigenvalues()[i].real > max_eigenvalue)
				{
					max_eigenvalue = solution.eigenvalues()[i].real;
					max_eigenvalue_idx = i;
				}
			}

			//Largest eigenvector
			Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType theta0_prev = theta0;
			theta0 = solution.eigenvectors.col(max_eigenvalue_idx);

			//Check if the eigenvector is sufficiently similar to the previous
			double dot = dot_amp(theta0_prev, theta0);
			double theta0_prev_size = 0.0, theta0_size = 0.0;
			for (int i = 0; i < 6; i++)
			{
				theta0_prev_size += theta0_prev[i].real()*theta0_prev[i].real() + 
					theta0_prev[i].imag()*theta0_prev[i].imag();
				theta0_size += theta0[i].real()*theta0[i].real() + theta0[i].imag()*theta0[i].imag();
			}

			//Cosine of the angle between the eigenvectors
			double cos_angle = dot / ( std::sqrt(theta0_prev_size) * std::sqrt(theta0_size) );

			//Ratio of smaller eigenvector amplitude to the larger eigenvector amplitude
			double amp_ratio = theta0_size > theta0_prev_size ? theta0_prev_size / theta0_size : theta0_size / theta0_prev_size;

			//Check if the eigenvectors of successive angles are sufficiently similar
			if (cos_angle * amp_ratio > thresh)
			{
				break;
			}
		}
	}
}