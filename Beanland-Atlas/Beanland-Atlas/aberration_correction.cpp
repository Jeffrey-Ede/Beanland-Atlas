#include <aberration_correction.h>

namespace ba
{
	/*Calculate the blurriness of an image using the variance of it's Laplacian filtrate. A 3 x 3 
	**{0, -1, 0; -1, 4, -1; 0 -1 0} Laplacian kernel is used. A two pass algorithm is used to avoid catastrophic
	**cancellation
	**Inputs:
	**img: cv::Mat &, 32-bit image to measure the blurriness of
	**mask: cv::Mat &, 8-bit mask where non-zero values indicate values of the Laplacian filtrate to use
	**Returns:
	**float, Variance of the Laplacian filtrate
	*/
	float var_laplacian(cv::Mat &img, cv::Mat &mask)
	{
		//Laplacian of image
		cv::Mat lap;
		cv::Laplacian(img, lap, CV_32FC1);

		//Calculate sample mean. A 2 pass algorithm is used to avoid catastrophic cancellation. Not that it is
		//super-necessary here as the mean of the laplacians is expected to be near zero...
		float mean = cv::mean(lap, mask)[0];

		//Calculate the variance of the Laplacian filtrate at the marked locations
		float sum_sqr_diff = 0.0f;
		int num_mask_px = 0;
		float *p;
		byte *b;
        #pragma omp parallel for reduction(sum:sum_sqr_diff), reduction(sum:num_mask_px)
		for (int i = 0; i < img.rows; i++)
		{
			p = lap.ptr<float>(i);
			b = mask.ptr<byte>(i);
			for (int j = 0; j < img.cols; j++)
			{
				//Check if the pixel is marked for calculation on the mask
				if (b[j])
				{
					sum_sqr_diff += (p[j] - mean)*(p[j] - mean);
					num_mask_px++;
				}
			}
		}

		return sum_sqr_diff / (num_mask_px - 1);
	}

	/*Use cluster analysis to label the high intensity pixels and produce a mask where their positions are marked. These positions
	**are indicative of edges
	**Inputs:
	**img: cv::Mat &, Image to find the high Scharr filtrate values of
	**dst: cv::Mat &, Output 8-bit image where the high Scharr filtrate values are marked
	**num_erodes: const int, Number of times to erode the mask to remove stray fluctating pixels
	**num_dilates: const int, Number of times to dilate the image after erosion
	**max_iter: const int, Maximum number of iterations to perform during k-means clustering
	**term_crit_eps: const float, Difference between successive k-means clustering iterations for the algorithm to stop
	**num_clusters: const int, Number of clusters to split data into to select the highest of. Defaults to 2.
	**val: const byte, Value to mark the high Scharr filtrate values with. Defaults to 1.
	*/
	void high_Scharr_edges(cv::Mat &img, cv::Mat &dst, const int num_erodes, const int num_dilates, const int max_iter,
		const float term_crit_eps, const int num_clusters, const byte val)
	{
		//Create the mask image if it is empty
		if (dst.empty())
		{
			dst = cv::Mat(img.size(), CV_8UC1);
		}

		//Get the amplitude of the image's Scharr filtrate
		cv::Mat scharr;
		scharr_amp(img, scharr);

		//Make a 1D copy of the Scharr filtrate for k-means clustering
		cv::Mat scharr1D = cv::Mat(img.rows*img.cols, 1, CV_32FC1);
		float *p;
        #pragma omp parallel for
		for( int y = 0; y < img.rows; y++ )
		{
			p = scharr.ptr<float>(y);
			for( int x = 0; x < img.cols; x++ )
			{ 
				scharr1D.at<float>(y + x*img.rows) = p[x];
			}
		}

		//Use k-means clustering to get the high Scharr filtrate
		cv::Mat labels, centers;
		cv::kmeans(
			scharr1D, 
			num_clusters,
			labels,
			cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, max_iter, term_crit_eps),
			1,
			cv::KMEANS_PP_CENTERS,
			centers 
		);

		//Construct the mask
		byte *b;
        #pragma omp parallel for
		for( int y = 0; y < img.rows; y++ )
		{
			b = dst.ptr<byte>(y);
			for( int x = 0; x < img.cols; x++ )
			{ 
				b[x] = centers.at<float>( labels.at<int>(y + x*img.rows, 0), 0 );
			}
		}

		//Erode the image to remove stray pixels
		erode(dst, dst, cv::Mat(), cv::Point(-1,-1), num_erodes);
		dilate(dst, dst, cv::Mat(), cv::Point(-1,-1), num_dilates);
	}
}