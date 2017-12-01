#include <beanland_atlas.h>

/*Find the positions of the spots in the aligned image average pattern. Most of these spots are made from the contributions of many images
**so it can be assumed that they are relatively featureless
**Inputs:
**align_avg: cv::Mat &, Average values of px in aligned diffraction patterns
**radius: int, Radius of spots
**thickness: int, Thickness of annulus to convolve with spots
**annulus_creator: cl_kernel, OpenCL kernel to create padded unblurred annulus to cross correlate the aligned image average pattern Sobel
**filtrate with
**circle_creator: cl_kernel, OpenCL kernel to create padded unblurred circle to cross correlate he aligned image average pattern with
**gauss_creator: cl_kernel, OpenCL kernel to create padded Gaussian to blur the annulus and circle with
**af_queue: cl_command_queue, ArrayFire command queue
**align_avg_cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
**align_avg_rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
**Return:
std::vector<cv::Point> Positions of spots in the aligned image average pattern
*/
std::vector<cv::Point> get_spot_pos(cv::Mat &align_avg, int radius, int thickness, cl_kernel annulus_creator, cl_kernel circle_creator, 
	cl_kernel gauss_creator, cl_command_queue af_queue, int align_avg_cols, int align_avg_rows)
{
	//Create vector to hold the spot positions
	std::vector<cv::Point> positions;

	//ArrayFire can only performs Fourier analysis on arrays that are a power of 2 in size so pad the input arrays to this size
	int cols = ceil_power_2(align_avg_cols);
	int rows = ceil_power_2(align_avg_rows);

	//Use an intermediate image to load the resized aligned average pixel values onto the GPU, otherwise memory is not contiguous
	cv::Mat contig_align_avg;
	cv::resize(align_avg, contig_align_avg, cv::Size(cols, rows), 0, 0, cv::INTER_LANCZOS4); //Resize the array so that it is a power of 2 in size
	af::array align_avg_af(cols, rows, (float*)contig_align_avg.data);

	//Fourier transform the aligned average pixel values
	af_array align_avg_fft_c;
	af_fft2_r2c(&align_avg_fft_c, align_avg_af.get(), 1.0f, cols, rows);

	//Refine the estimate for the spot separation using the aligned images average px values
	/*TEMP*/int ubound = 130;

	//Create Fourier transform of the Gaussian to blur the annulus and circle with, remembering that ArrayFire and OpenCV arrays are transpositional
	af::array padded_gauss = extended_gauss(rows, cols, 0.25*UBOUND_GAUSS_SIZE+0.75, gauss_creator, af_queue);

	//Fourier transform the Gaussian
	af_array gauss_fft_c;
	af_fft2_r2c(&gauss_fft_c, padded_gauss.get(), 1.0f, cols, rows);
	af::array gauss_fft = af::array(gauss_fft_c);

	//Create circle to cross correlate the aligned average pixel values with
	af::array circle = create_circle(cols*rows, rows, rows/2, cols, cols/2, radius, circle_creator, af_queue);

	//Fourier transform the circle
	af_array circle_fft_c;
	af_fft2_r2c(&circle_fft_c, circle.get(), 1.0f, cols, rows);
	af::array circle_fft = af::array(circle_fft_c);

	//Gaussian blur the circle in the Fourier domain and cross correlate it with the Fourier transform of the aligned average pixel values
	af_array circle_xcorr;
	af_fft2_c2r(&circle_xcorr, (1e-10 * gauss_fft*circle_fft*af::array(align_avg_fft_c)).get(), 1.0f, false);

	//Create annulus to cross correlate the aligned average pixel values with
	af::array annulus = create_annulus(rows*cols, rows, rows/2, cols, cols/2, radius, thickness, annulus_creator, af_queue);

	//Fourier transform the annulus
	af_array annulus_fft_c;
	af_fft2_r2c(&annulus_fft_c, annulus.get(), 1.0f, cols, rows);
	af::array annulus_fft = af::array(annulus_fft_c);

	//Fourier transform the image's Sobel filtrate
	af_array sobel_filtrate_fft_c;
	af_fft2_r2c(&sobel_filtrate_fft_c, af::sobel(align_avg_af, SOBEL_SIZE, false).get(), 1.0f, cols, rows);

	//Gaussian blur the annulus in the Fourier domain and cross correlate it with the Fourier transform of the aligned average pixel values
	//Sobel filtrate
	af_array annulus_xcorr;
	af_fft2_c2r(&annulus_xcorr, (1e-10 * gauss_fft*annulus_fft*af::array(sobel_filtrate_fft_c)).get(), 1.0f, false);

	//Transfer the product of the circular and annular cross correlations back to the host
	float *xcorr_data = (af::array(circle_xcorr)*af::array(annulus_xcorr)).host<float>();

	//Modify the dimensions of the 1D array returned from the device to the host so that it is the correct size
	cv::Mat xcorr = cv::Mat(cols, rows, CV_32FC1, xcorr_data);

	//Find the location of the maximum cross correlation. This gives the position of the brightest spot and therefore the center of the
	//diffraction pattern
	cv::Point maxLoc;
	cv::minMaxLoc(xcorr, NULL, NULL, NULL, &maxLoc);

	//Store the position of the maximum
	positions.push_back(maxLoc);

	//Assume spots are squarely packed as this gives the lowest packing densisty. Black out spots until a set proportion of the spots
	// have been blacked out under this assumption
	const int search_num = (int)(BLACKOUT_PROP * rows*cols / (ubound*ubound));
	for (int i = 0; i < search_num; i++) 
	{
		//Blacken the brightest spot and find the next brightest
		blacken_circle(contig_align_avg, maxLoc.x, maxLoc.y, ubound/2);

		//Load the blackened image onto the GPU
		af::array blackened_af(cols, rows, (float*)contig_align_avg.data);

		/* Repeat the Fourier analysis to find the next brightest spot */
		//Fourier transform the Sobel filtrate
		af_fft2_r2c(&sobel_filtrate_fft_c, af::sobel(blackened_af, SOBEL_SIZE, false).get(), 1.0f, cols, rows);

		//Cross correlate the Sobel filtrate of the blackened image with the annulus
		af_fft2_c2r(&annulus_xcorr, (1e-10 * gauss_fft*annulus_fft*af::array(sobel_filtrate_fft_c)).get(), 1.0f, false);

		//Fourier transform the blackened image
		af_fft2_r2c(&align_avg_fft_c, blackened_af.get(), 1.0f, cols, rows);

		//Cross correlate the image with the circle
		af_fft2_c2r(&circle_xcorr, (1e-10 * gauss_fft*circle_fft*af::array(align_avg_fft_c)).get(), 1.0f, false);

		//Transfer the product of the annulus and circle cross correlations back to the host
		xcorr_data = (af::array(circle_xcorr)*af::array(annulus_xcorr)).host<float>();

		//Modify the dimensions of the 1D array returned from the device to the host so that it is the correct size
		cv::Mat xcorr = cv::Mat(cols, rows, CV_32FC1, xcorr_data);

		//Find the location of the maximum cross correlation. This gives the position of the brightest spot and therefore the center of the
		//diffraction pattern
		cv::Point maxLoc;
		cv::minMaxLoc(xcorr, NULL, NULL, NULL, &maxLoc);

		//Store the position of the maximum
		positions.push_back(maxLoc);
	}

	//Free memory
	free(xcorr_data);

	positions[0] = cv::Point(4, 5);
	return positions;
}

/*Blackens the circle of pixels within a certain radius of a point in a floating point OpenCV mat
**Inputs:
**mat: cv::Mat &, Reference to a floating point OpenCV mat to blacken a circle on
**col: const int, column of circle origin
**row: const int, row of circle origin
**rad: const int, radius of the circle to blacken
*/
void blacken_circle(cv::Mat &mat, const int col, const int row, const int rad)
{
	//Get the minimum and maximum rows and columns to iterate between
	int min_col = std::max(0, col-rad);
	int max_col = std::min(mat.cols-1, col+rad);
	int min_row = std::max(0, row-rad);
	int max_row = std::min(mat.rows-1, row+rad);

	//Iterate accross the circle rows
	float *p;
    #pragma omp parallel for
	for (int i = min_row, rel_row = -rad; i <= max_row; i++, rel_row++)
	{
		//Create C style pointer to interate across the circle with
		p = mat.ptr<float>(i);

		//Get columns to iterate between
		int c = (int)std::sqrt(rad*rad-rel_row*rel_row);
		int min = std::max(min_col, col-c);
		int max = std::min(max_col, col+c);

		//Iterate across columns
		for (int j = min; j <= max; j++)
		{
			p[j] = 0.0f;
		}
	}
}