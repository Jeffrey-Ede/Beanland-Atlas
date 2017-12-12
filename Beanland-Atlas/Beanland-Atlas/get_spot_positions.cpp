#include <beanland_atlas.h>

namespace ba
{
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
		const int search_num = rows*cols / (ubound*ubound);
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
			cv::minMaxLoc(xcorr, NULL, NULL, NULL, &maxLoc);

			//Store the position of the maximum
			positions.push_back(maxLoc);
		}

		//Extract the lattice vectors
		std::vector<cv::Vec2i> lattice_vectors = get_lattice_vectors(positions);

		//Remove or correct any outlier spots
		std::vector<cv::Point> on_latt_spots = correct_spot_pos(positions, lattice_vectors, align_avg_cols, align_avg_rows, radius);

		////Refine the lattice vectors
		//std::vector<cv::Vec2f> refined_latt_vect = refine_lattice_vectors(on_latt_spots, lattice_vectors);

		//Use the lattice vectors to find additional spots in the aligned images average px values pattern
		find_other_spots(xcorr, on_latt_spots, lattice_vectors, align_avg_cols, align_avg_rows, radius);

		//Rescale the spot positions to the origninal dimensions of the aligned average image
        #pragma omp parallel for
		for (int i = 0; i < on_latt_spots.size(); i++)
		{
			on_latt_spots[i].x = (on_latt_spots[i].x * align_avg_cols) / cols;
			on_latt_spots[i].y = (on_latt_spots[i].y * align_avg_rows) / rows;
		}

		//Free memory
		free(xcorr_data);

		return on_latt_spots;
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

	/*Uses a set of know spot positions to extract approximate lattice vectors for a diffraction pattern
	**Input:
	**positions: std::vector<cv::Point> &, Known positions of spots in the diffraction pattern
	**Returns:
	**std::vector<cv::Vec2f>, The lattice vectors
	*/
	std::vector<cv::Vec2i> get_lattice_vectors(std::vector<cv::Point> &positions)
	{
		//Assign memory to store the lattice vectors
		std::vector<cv::Vec2i> lattice_vectors(2);

		float min_pair_sep = INT_MAX; //A very large value so that first comparison will be stored as the minimum separation
		int min_pair_dx;
		int min_pair_dy;

		//Get the relative positions of all the spots relative to the brightest and find te minimum
		for (int i = 1; i < positions.size(); i++)
		{
			//Get differences in position
			int dx = positions[i].x - positions[0].x;
			int dy = positions[i].y - positions[0].y;

			//If this is the minimum separation so far, record its parameters
			if (std::sqrt(dx*dx + dy*dy) < min_pair_sep)
			{
				min_pair_sep = std::sqrt(dx*dx + dy*dy);
				min_pair_dx = dx;
				min_pair_dy = dy;
			}
		}

		//Store the difference between the minimally separated pair to the return vector
		lattice_vectors[0] = cv::Vec2i(min_pair_dx, min_pair_dy);

		//Get angle between the first lattice vector and a vector going straight accross the matrix
		float angle = std::acos((float)min_pair_dx / min_pair_sep); //Angle will be between 0 and pi	

		//Look for the minimum vector that is pointing in a significantly different direction
		min_pair_sep = INT_MAX; //A very large value so that first comparison will be stored as the minimum separation

		//Get the relative positions of all the spots relative to the brightest and find the minimum that is at least some angle different
		//from the first lattice vector found
		for (int i = 1; i < positions.size(); i++)
		{
			//Get differences in position
			int dx = positions[i].x - positions[0].x;
			int dy = positions[i].y - positions[0].y;

			//Separation
			float sep = std::sqrt(dx*dx + dy*dy);

			//Consider recording if this is the minimum separation so far
			if (sep < min_pair_sep)
			{
				float angle2 = std::acos(dx / sep);
				
				//If the angle of this lattice vector is sufficiently different to the previous, record it
				if (std::abs(angle2 - angle) > LATTICE_VECT_DIR_DIFF)
				{
					min_pair_sep = std::sqrt(dx*dx + dy*dy);
					min_pair_dx = dx;
					min_pair_dy = dy;
				}
			}
	}

		//Store the second lattice vector
		lattice_vectors[1] = cv::Vec2i(min_pair_dx, min_pair_dy);

		return lattice_vectors;
	}

	/*Uses lattice vectors to search for spots in the diffraction pattern that have not already been recorded
	**Input:
	**xcorr: cv::Mat &, Product of annulus and circle cross correlations that has been blackened where spots have already been found
	**positions: std::vector<cv::Point> &, Known positions of spots in the diffraction pattern
	**lattice_vectors: std::vector<cv::Vec2i> &, Lattice vectors describing the positions of the spots
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rad: int, radius of the spots
	*/
	void find_other_spots(cv::Mat &xcorr, std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &lattice_vectors, 
		int cols, int rows, int rad)
	{
		//Search specified area around lattice point
		const int search_radius = std::min((int)(SCALE_SEARCH_RAD*rad), 1);

		//Calculate the maximum and minimum multiples of the lattice vectors that are in the image
		int max_vect1 = 1;
		int min_vect1 = -1;
		int max_vect2 = 1;
		int min_vect2 = -1;

		#pragma omp parallel sections
		{
			#pragma omp section
			{
				//Calculate the maximum multiple of the first lattice vector from the central spot that is in range
				while (max_vect1*lattice_vectors[0][0] + positions[0].x < cols && max_vect1*lattice_vectors[0][1] + positions[0].y < rows &&
					max_vect1*lattice_vectors[0][0] + positions[0].x >= 0 && max_vect1*lattice_vectors[0][1] + positions[0].y >= 0)
				{
					max_vect1++;
				}
			}

			#pragma omp section
			{
				//Calculate the minimum multiple of the first lattice vector from the central spot that is in range
				while (min_vect1*lattice_vectors[0][0] + positions[0].x < cols && min_vect1*lattice_vectors[0][1] + positions[0].y < rows &&
					min_vect1*lattice_vectors[0][0] + positions[0].x >= 0 && min_vect1*lattice_vectors[0][1] + positions[0].y >= 0)
				{
					min_vect1--;
				}
			}

			#pragma omp section
			{
				//Calculate the maximum multiple of the second lattice vector from the central spot that is in range
				while (max_vect2*lattice_vectors[1][0] + positions[0].x < cols && max_vect2*lattice_vectors[1][1] + positions[0].y < rows && 
					max_vect2*lattice_vectors[1][0] + positions[0].x >= 0 && max_vect2*lattice_vectors[1][1] + positions[0].y >= 0) 
				{
					max_vect2++;
				}
			}

			#pragma omp section
			{
				//Calculate the minimum multiple of the second lattice vector from the central spot that is in range
				while (min_vect2*lattice_vectors[1][0] + positions[0].x < cols && min_vect2*lattice_vectors[1][1] + positions[0].y < rows && 
					min_vect2*lattice_vectors[1][0] + positions[0].x >= 0 && min_vect2*lattice_vectors[1][1] + positions[0].y >= 0) 
				{
					min_vect2--;
				}
			}
		}

		//Iterate across multiples of the first lattice vector
		for (int i = min_vect1; i <= max_vect1; i++)
		{
			//Iterate across multiples of the second lattice vector
			for (int j = min_vect2; j <= max_vect2; j++)
			{
				//Calculate the location of this combination of lattice vectors
				float col = i*lattice_vectors[0][0] + j*lattice_vectors[1][0] + positions[0].x;
				float row = i*lattice_vectors[0][1] + j*lattice_vectors[1][1] + positions[0].y;

				//Check that a spot has not already been located in in the region
				bool pos_not_loc = true;
				for (int k = 0; k < positions.size(); k++)
				{
					if (std::sqrt((col - positions[k].x)*(col - positions[k].x) + (row - positions[k].y)*(row - positions[k].y)) <= search_radius)
					{
						pos_not_loc = false;
						break;
					}
				}

				//If the lattice point is in the image
				if (pos_not_loc && row >= 0 && row < rows && col >= 0 && col < cols)
				{
					//Store the position
					positions.push_back(cv::Point(row, col));
				}
			}
		}
	}

	/*Remove or correct any spot positions that do not fit on the spot lattice very well
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots
	**lattice_vectors: std::vector<cv::Vec2i> &, Initial lattice vector estimate
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rad: int, radius of the spots
	**tol: float, The maximum fraction of the circle radius that a circle can be away from a multiple of the initial
	**lattice vectors estimate and still be considered a valid spot
	**Returns:
	**std::vector<cv::Point>, Spots that lie within tolerance of the lattice defined by the lattice vectors
	*/
	std::vector<cv::Point> correct_spot_pos(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &lattice_vectors,	int cols, int rows,
		int rad, float tol)
	{

		//Maximum tolerated difference from expected lattice positions
		float max_diff = tol * rad;

		//Record whether the spot positions, besides that of the central spot, agree with the lattice vectors
		std::vector<bool> on_latt(positions.size()-1, false);

		//Calculate the maximum and minimum multiples of the lattice vectors that are in the image
		int max_vect1 = 1;
		int min_vect1 = -1;
		int max_vect2 = 1;
		int min_vect2 = -1;

        #pragma omp parallel sections
		{
            #pragma omp section
			{
				//Calculate the maximum multiple of the first lattice vector from the central spot that is in range
				while (max_vect1*lattice_vectors[0][0] + positions[0].x < cols && max_vect1*lattice_vectors[0][1] + positions[0].y < rows &&
					max_vect1*lattice_vectors[0][0] + positions[0].x >= 0 && max_vect1*lattice_vectors[0][1] + positions[0].y >= 0)
				{
					max_vect1++;
				}
			}

            #pragma omp section
			{
				//Calculate the minimum multiple of the first lattice vector from the central spot that is in range
				while (min_vect1*lattice_vectors[0][0] + positions[0].x < cols && min_vect1*lattice_vectors[0][1] + positions[0].y < rows &&
					min_vect1*lattice_vectors[0][0] + positions[0].x >= 0 && min_vect1*lattice_vectors[0][1] + positions[0].y >= 0)
				{
					min_vect1--;
				}
			}

            #pragma omp section
			{
				//Calculate the maximum multiple of the second lattice vector from the central spot that is in range
				while (max_vect2*lattice_vectors[1][0] + positions[0].x < cols && max_vect2*lattice_vectors[1][1] + positions[0].y < rows && 
					max_vect2*lattice_vectors[1][0] + positions[0].x >= 0 && max_vect2*lattice_vectors[1][1] + positions[0].y >= 0) 
				{
					max_vect2++;
				}
			}

            #pragma omp section
			{
				//Calculate the minimum multiple of the second lattice vector from the central spot that is in range
				while (min_vect2*lattice_vectors[1][0] + positions[0].x < cols && min_vect2*lattice_vectors[1][1] + positions[0].y < rows && 
					min_vect2*lattice_vectors[1][0] + positions[0].x >= 0 && min_vect2*lattice_vectors[1][1] + positions[0].y >= 0) 
				{
					min_vect2--;
				}
			}
		}

		//Iterate across multiples of the first lattice vector
		int num_on_latt = 0;
		for (int i = min_vect1; i <= max_vect1; i++)
		{
			//Iterate across multiples of the second lattice vector
			for (int j = min_vect2; j <= max_vect2; j++)
			{
				//Calculate the location of this combination of lattice vectors
				int col = i*lattice_vectors[0][0] + j*lattice_vectors[1][0] + positions[0].x;
				int row = i*lattice_vectors[0][1] + j*lattice_vectors[1][1] + positions[0].y;

				//If a spot is within tolerance of this lattice vector combination, mark it as being on the lattice
				for (int k = 1; k < positions.size(); k++)
				{
					if (std::sqrt((col - positions[k].x)*(col - positions[k].x) + (row - positions[k].y)*(row - positions[k].y)) <= max_diff)
					{
						on_latt[k-1] = true;
						num_on_latt++;
					}
				}
			}
		}

		//Return the on lattice spot positions
		std::vector<cv::Point> on_latt_spots(num_on_latt);
		on_latt_spots[0] = positions[0]; //The brightest spot is on the lattice
		for (int i = 1, j = 0; i < positions.size(); i++, j++)
		{
			//If the spot is on the lattice
			if (on_latt[j])
			{
				on_latt_spots[i] = positions[i];
			}
		}

		return on_latt_spots;
	}

	/*Refine the lattice vectors
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**est_latt_vect: std::vector<cv::Vec2i> &, Original estimate of the lattice vectors
	**Returns:
	**std::vector<cv::Vec2f> &, Refined estimate of the lattice vectors
	*/
	std::vector<cv::Vec2f> refine_lattice_vectors(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &est_latt_vect)
	{
		//To be implemented. Possibly by studying the position of the maxima of the Fourier transform of the lattice as it is vibrated

		std::vector<cv::Vec2f> latt_vect;
		return latt_vect;
	}
}