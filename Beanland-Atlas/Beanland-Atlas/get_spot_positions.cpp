#include <beanland_atlas.h>

/*Find the positions of the spots in the aligned image average pattern. Most of these spots are made from the contributions of many images
**so it can be assumed that they are relatively featureless
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
std::vector<std::array<int, 2>> Positions of spots in the aligned image average pattern
*/
std::vector<std::array<int, 2>> get_spot_pos(cv::Mat &align_avg, int radius, int thickness, cl_kernel annulus_creator, cl_kernel circle_creator, 
	cl_kernel gauss_creator, cl_command_queue af_queue, int align_avg_cols, int align_avg_rows)
{
	//Create vector to hold the spot positions
	std::vector<std::array<int, 2>> positions;

	//ArrayFire r2c FFt only accepts arrays with even dimensions so truncate arrays by 1 row or column to make their numbers even, if necessary
	int cols = align_avg_cols - align_avg_cols%2;
	int rows = align_avg_rows - align_avg_rows%2;

	//Use an intermediate image to load the truncated aligned average pixel values onto the GPU, otherwise memory is not contiguous
	cv::Rect roi(0, 0, rows, cols);
	cv::Mat contig_align_avg;
	align_avg(roi).convertTo(contig_align_avg, CV_32FC1, 1);
	af::array align_avg_af(rows, cols, (float*)contig_align_avg.data);

	//Create Fourier transform of the Gaussian to blur the annulus and circle with, remembering that ArrayFire and OpenCV arrays are transpositional
	af::array padded_gauss = extended_gauss(rows, cols, 0.25*UBOUND_GAUSS_SIZE+0.75, gauss_creator, af_queue);

	//Fourier transform the Gaussian
	af_array gauss_fft_c;
	af_fft2_r2c(&gauss_fft_c, padded_gauss.get(), 1.0f, cols, rows);
	af::array gauss_fft = af::array(gauss_fft_c);

	af::print("af", gauss_fft);

	//Create annulus to cross correlate the aligned average pixel values with
	af::array annulus = create_annulus(rows*cols, rows, rows/2, cols, cols/2, radius, thickness, annulus_creator, af_queue);

	//Fourier transform the annulus
	af_array annulus_fft_c;
	af_fft2_r2c(&annulus_fft_c, annulus.get(), 1.0f, cols, rows);

	//Fourier transform the image's Sobel filtrate
	af_array sobel_filtrate_fft_c;
	af_fft2_r2c(&sobel_filtrate_fft_c, af::sobel(align_avg_af, SOBEL_SIZE, false).get(), 1.0f, cols, rows);

	//Gaussian blur the annulus in the Fourier domain and cross correlate it with the Fourier transform of the aligned average pixel values
	//Sobel filtrate
	af_array annulus_xcorr;
	af_fft2_c2r(&annulus_xcorr, (1e-10 * af::array(gauss_fft_c)*af::array(annulus_fft_c)*af::array(sobel_filtrate_fft_c)).get(), 1.0f, false);

	//Create circle to cross correlate the aligned average pixel values with
	af::array circle = create_circle(cols*rows, rows, rows/2, cols, cols/2, radius, annulus_creator, af_queue);

	//Fourier transform the circle
	af_array circle_fft_c;
	af_fft2_r2c(&circle_fft_c, circle.get(), 1.0f, cols, rows);

	//Fourier transform the aligned average pixel values
	af_array align_avg_fft_c;
	af_fft2_r2c(&align_avg_fft_c, align_avg_af.get(), 1.0f, cols, rows);

	//Gaussian blur the circle in the Fourier domain and cross correlate it with the Fourier transform of the aligned average pixel values
	af_array circle_xcorr;
	af_fft2_c2r(&circle_xcorr, (1e-10 * af::array(gauss_fft_c)*af::array(circle_fft_c)*af::array(align_avg_fft_c)).get(), 1.0f, false);

	if(1){
		//const static int width = size, height = size;
		af::Window window(512, 512, "2D plot");
		do{
			int num_acc_px = 1000;
			window.image(af::array(circle_xcorr).as(f32)/(5e2));
		} while( !window.close() );
	}

	//Transfer the product of the circular and annular cross correlations back to the host
	float *xcorr_data = (af::array(circle_xcorr)*af::array(annulus_xcorr)).host<float>();

	//Modify the dimensions of the 1D array returned from the device to the host so that it is the correct size
	cv::Mat xcorr = cv::Mat(cols, rows, CV_32FC1, xcorr_data);

	imshow("af", xcorr/1000);

	cv::waitKey(0);

	//Record maxima until ***

	positions[0] = {1, 2};
	return positions;
}
