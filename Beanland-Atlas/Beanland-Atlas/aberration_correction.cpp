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
	*/
	void high_Scharr_edges(cv::Mat &img, cv::Mat &dst, const int num_erodes, const int num_dilates, const int max_iter,
		const float term_crit_eps, const int num_clusters)
	{
		//Create the mask image if it is empty
		if (dst.empty())
		{
			dst = cv::Mat(img.size(), CV_8UC1);
		}

		//Get the amplitude of the image's Scharr filtrate
		cv::Mat scharr;
		scharr_amp(img, scharr);

		std::vector<int> clusters_to_use(1, -1);
		kmeans_mask(scharr, dst, num_clusters, clusters_to_use, cv::Mat(), max_iter, term_crit_eps);

		//Erode the image to remove stray pixels
		cv::erode(dst, dst, cv::Mat(), cv::Point(-1,-1), num_erodes);

		//Compensate for the erosion and/or slightly pad the region around pixels of high gradiation
		cv::dilate(dst, dst, cv::Mat(), cv::Point(-1,-1), num_dilates);
	}

	/*Mark intensity groups in a single channel image using k-means clustering
	**Inputs:
	**img: cv::Mat &, The single-channel 32-bit image to posterise
	**dst: cv::Mat &, 8-bit output image
	**num_clusters: const int, Number of groups to split the intensities into
	**clusters_to_use: std::vector<int>, Colour quantisation levels to use. Clusters can be indicated from low to high using 
	**numbers starting from 0, going up. Clusters can be indicated from high to low using numbers starting from -1, going down
	**mask: cv::Mat &, Optional 8-bit image whose non-zero values are to be k-means clustered. Zero values on the mask will be zeroes
	**in the output image
	**max_iter: const int, Maximum number of iterations to perform during k-means clustering
	**term_crit_eps: const float, Difference between successive k-means clustering iterations for the algorithm to stop
	**val: const byte, Value to mark values to use on the output image. Defaults to 1
	*/
	void kmeans_mask(cv::Mat &img, cv::Mat &dst, const int num_clusters, std::vector<int> clusters_to_use, cv::Mat &mask, 
		const int max_iter, const float term_crit_eps, const byte val)
	{
		//Convert negative intensity cluster values to their positive counterparts
		for (int i = 0; i < clusters_to_use.size(); i++)
		{
			if (clusters_to_use[i] < 0)
			{
				clusters_to_use[i] += clusters_to_use.size();
			}
		}

		if (mask.empty())
		{
			//Declare output mask, not initialising values
			dst = cv::Mat(img.size(), CV_8UC1);

			//Make a 1D copy of the image for k-means clustering
			cv::Mat img1D;
			if (img.cols > 1)
			{
				img1D= cv::Mat(img.rows*img.cols, 1, CV_32FC1);
				float *p;
                #pragma omp parallel for
				for( int y = 0; y < img.rows; y++ )
				{
					p = img.ptr<float>(y);
					for( int x = 0; x < img.cols; x++ )
					{ 
						img1D.at<float>(y + x*img.rows) = p[x];
					}
				}
			}
			else
			{
				img1D = img;
			}

			//Perform k-means clustering
			cv::Mat labels, centers;
			cv::kmeans(
				img1D, 
				num_clusters,
				labels,
				cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, max_iter, term_crit_eps),
				1,
				cv::KMEANS_PP_CENTERS,
				centers 
			);

			//Sort the cluster centres to find out which ones are to be marked
			std::vector<float> sorted_centers(centers);
			std::sort(sorted_centers.begin(), sorted_centers.end());

			//Record the centers to be used
			std::vector<float> centers_to_use(clusters_to_use.size());
			for (int i = 0; i < clusters_to_use.size(); i++)
			{
				if (contains(clusters_to_use, i))
				{
					centers_to_use[i] = sorted_centers[i];
				}
			}

			//Construct the mask
			byte *b;
            #pragma omp parallel for
			for( int y = 0; y < img.rows; y++ )
			{
				b = dst.ptr<byte>(y);
				for( int x = 0; x < img.cols; x++ )
				{ 
					b[x] = contains(centers_to_use, centers.at<float>( labels.at<int>(y + x*img.rows, 0), 0 )) ? val : 0;
				}
			}
		}
		else
		{
			//Declare output mask, not initialising values
			dst = cv::Mat(img.size(), CV_8UC1, cv::Scalar(0));

			//Make a 1D copy of the marked image pixels for k-means clustering
			int nnz_px = cv::countNonZero(img);
			cv::Mat img1D = cv::Mat(nnz_px, 1, CV_32FC1);
			float *p;
			byte *m;
			for( int y = 0, k = 0; y < img.rows; y++ )
			{
				m = mask.ptr<byte>(y);
				p = img.ptr<float>(y);
				for( int x = 0; x < img.cols; x++)
				{ 
					//Add pixels that are marked on the mask
					if (m[x])
					{
						img1D.at<float>(k) = p[x];
						k++;
					}
				}
			}

			//Perform k-means clustering
			cv::Mat labels, centers;
			cv::kmeans(
				img1D, 
				num_clusters,
				labels,
				cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, max_iter, term_crit_eps),
				1,
				cv::KMEANS_PP_CENTERS,
				centers 
			);

			//Sort the cluster centres to find out which ones are to be marked
			std::vector<float> sorted_centers(centers);
			std::sort(sorted_centers.begin(), sorted_centers.end());

			//Record the centers to be used
			std::vector<float> centers_to_use(clusters_to_use.size());
			for (int i = 0; i < clusters_to_use.size(); i++)
			{
				if (contains(clusters_to_use, i))
				{
					centers_to_use[i] = sorted_centers[i];
				}
			}

			byte *b;
			for( int y = 0, k = 0; y < img.rows; y++ )
			{
				m = mask.ptr<byte>(y);
				b = dst.ptr<byte>(y);
				for( int x = 0; x < img.cols; x++ )
				{ 
					//Add pixels that are marked on the mask
					if (m[x])
					{
						b[x] = contains(centers_to_use, centers.at<float>( labels.at<int>(k, 0), 0 )) ? val : 0;
						k++;
					}
				}
			}
		}
	}

	/*Mark a single intensity group in a single channel image using k-means clustering. This function passes a vector indicating the
	**single intensity group to the variant of the function that accepts multiple intensity groups
	**Inputs:
	**img: cv::Mat &, The single-channel 32-bit image to posterise
	**dst: cv::Mat &, 8-bit output image
	**num_clusters: const int, Number of groups to split the intensities into
	**cluster_to_use: const int, Colour quantisation level to use. Clusters can be indicated from low to high using 
	**numbers starting from 0, going up. Clusters can be indicated from high to low using numbers starting from -1, going down
	**mask: cv::Mat &, Optional 8-bit image whose non-zero values are to be k-means clustered. Zero values on the mask will be zeroes
	**in the output image
	**max_iter: const int, Maximum number of iterations to perform during k-means clustering
	**term_crit_eps: const float, Difference between successive k-means clustering iterations for the algorithm to stop
	**val: const byte, Value to mark values to use on the output image. Defaults to 1
	*/
	void kmeans_mask(cv::Mat &img, cv::Mat &dst, const int num_clusters, const int cluster_to_use, cv::Mat &mask,
		const int max_iter, const float term_crit_eps, const byte val)
	{
		std::vector<int> temp(1, cluster_to_use);
		kmeans_mask(img, dst, num_clusters, temp, mask, max_iter, term_crit_eps, val);
	}
}