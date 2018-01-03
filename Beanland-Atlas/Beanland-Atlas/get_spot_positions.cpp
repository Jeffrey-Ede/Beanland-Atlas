#include <get_spot_positions.h>

namespace ba
{
	/*Find the positions of the spots in the aligned image average pattern. Most of these spots are made from the contributions of many images
	**so it can be assumed that they are relatively featureless
	**Inputs:
	**align_avg: cv::Mat &, Average values of px in aligned diffraction patterns
	**initial_radius: int, Radius of the spots
	**initial_thickness: int, Thickness of the annulus to convolve with spots
	**annulus_creator: cl_kernel, OpenCL kernel to create padded unblurred annulus to cross correlate the aligned image average pattern Sobel
	**filtrate with
	**circle_creator: cl_kernel, OpenCL kernel to create padded unblurred circle to cross correlate he aligned image average pattern with
	**gauss_creator: cl_kernel, OpenCL kernel to create padded Gaussian to blur the annulus and circle with
	**af_queue: cl_command_queue, ArrayFire command queue
	**align_avg_cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**align_avg_rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**samp_to_detect_sphere: cv::Vec2f &, Reference to store the sample-to-detector sphere radius and orientation estimated by this function
	**discard_outer: const int, Discard spots within this distance from the boundary. Defaults to discarding spots within 1 radius
	**Return:
	std::vector<cv::Point> Positions of spots in the aligned image average pattern
	*/
	std::vector<cv::Point> get_spot_pos(cv::Mat &align_avg, int initial_radius, int initial_thickness, cl_kernel annulus_creator,
		cl_kernel circle_creator, cl_kernel gauss_creator, cl_command_queue af_queue, int align_avg_cols, int align_avg_rows, 
		cv::Vec2f &samp_to_detect_sphere, const int discard_outer)
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

		//Approximately resize the annulus and circle parameters if the pattern was resized
		int radius, thickness;
		if (rows != align_avg_rows || cols != align_avg_cols)
		{
			//Approximately rescale the circle and annulus parameters
			float scale_factor = std::sqrt((cols/align_avg_cols)*(cols/align_avg_cols) + (rows/align_avg_rows)*(rows/align_avg_rows));

			radius = (int)(scale_factor*initial_radius);
			thickness = (int)(scale_factor*initial_thickness);
		}
		//If the aligned diffraction pattern did not need to be resized
		else
		{
			radius = initial_radius;
			thickness = initial_thickness;
		}

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
		std::vector<cv::Point> on_latt_spots = correct_spot_pos(positions, lattice_vectors, cols, rows, radius);

		//Refine the lattice vectors
		float latt_vect_range = LATT_REF_RANGE * std::max( std::sqrt(lattice_vectors[0][0]*lattice_vectors[0][0] +
			lattice_vectors[0][1]*lattice_vectors[0][1]), std::sqrt(lattice_vectors[1][0]*lattice_vectors[1][0] +
				lattice_vectors[1][1]*lattice_vectors[1][1]) );
		latt_vect_range = std::min(MIN_LATT_REF_RANGE, latt_vect_range);
		std::vector<cv::Vec2f> refined_latt_vect = refine_lattice_vectors(on_latt_spots, lattice_vectors,  cols, rows, 
			latt_vect_range, LATT_REF_REQ_ACC);

		//Use the lattice vectors to find additional spots in the aligned images average px values pattern
		find_other_spots(on_latt_spots, refined_latt_vect, cols, rows, radius);

		//Rescale the spot positions to the origninal dimensions of the aligned average image
        #pragma omp parallel for
		for (int i = 0; i < on_latt_spots.size(); i++)
		{
			on_latt_spots[i].x = (on_latt_spots[i].x * align_avg_cols) / cols;
			on_latt_spots[i].y = (on_latt_spots[i].y * align_avg_rows) / rows;
		}

		//Discard spots that are not at least a specified distance from the peripheries of the image
		positions = discard_outer_spots(on_latt_spots, cols, rows, discard_outer == -1 ? initial_radius : discard_outer);

		//Estimate the parameters decribing the sample-to-detector sphere
		samp_to_detect_sphere = get_sample_to_detector_sphere(on_latt_spots, xcorr, discard_outer == -1 || discard_outer >= initial_radius ? 0 : initial_radius,
			cols, rows);

		//Return original positions for now - something wrong with lattice vector refinement and I don't feel I have time to fix it
        #pragma omp parallel for
		for (int i = 0; i < on_latt_spots.size(); i++)
		{
			positions[i].x = (positions[i].x * align_avg_cols) / cols;
			positions[i].y = (positions[i].y * align_avg_rows) / rows;
		}

		//Free memory
		free(xcorr_data);

		return positions /*on_latt_spots*/;
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
	**positions: std::vector<cv::Point> &, Known positions of spots in the diffraction pattern
	**lattice_vectors: std::vector<cv::Vec2f> &, Lattice vectors describing the positions of the spots
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rad: int, radius of the spots
	*/
	void find_other_spots(std::vector<cv::Point> &positions, std::vector<cv::Vec2f> &lattice_vectors, 
		int cols, int rows, int rad)
	{
		//Search specified area around lattice point
		int search_radius = SCALE_SEARCH_RAD * std::min( std::sqrt(lattice_vectors[0][0]*lattice_vectors[0][0] +
			lattice_vectors[0][1]*lattice_vectors[0][1]), std::sqrt(lattice_vectors[1][0]*lattice_vectors[1][0] +
				lattice_vectors[1][1]*lattice_vectors[1][1]) );
		search_radius = std::max(search_radius, 1);

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
					positions.push_back(cv::Point((int)row, (int)col));
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

				//Only record the closest spot
				float min_dist = INT_MAX;
				int prev_k = 0;

				//If a spot is within tolerance of this lattice vector combination and is the closest spot, mark it as being on the lattice
				for (int k = 1; k < positions.size(); k++)
				{
					//Distance from the lattice point
					float dist = std::sqrt((col - positions[k].x)*(col - positions[k].x) + (row - positions[k].y)*(row - positions[k].y));

					//If it is closer than the minimum distance and is the closest to the spot
					if (dist <= max_diff && dist < min_dist)
					{
						//Record parameters of the spot
						on_latt[k-1] = true; //Mark that the position is on the lattice, within tolerance
						min_dist = dist; //The distance of the position from the lattice point to compare followubg positions against to see if they are closer

						//If this spot is closer than a previous spot, unmark the previous
						if (prev_k)
						{
							on_latt[prev_k-1] = false;
						}
						//If this is the first spot within range to be found
						else
						{
							num_on_latt++;
						}

						//Record the index of this spot in case a future spot is closer
						prev_k = k;
					}
				}
			}
		}

		//Return the on lattice spot positions
		std::vector<cv::Point> on_latt_spots(num_on_latt);
		on_latt_spots[0] = positions[0]; //The brightest spot is on the lattice
		for (int i = 1, j = 1; i < positions.size(); i++)
		{
			//If the spot is on the lattice
			if (on_latt[i])
			{
				on_latt_spots[j] = positions[i];
				j++;
			}
		}

		return on_latt_spots;
	}

	/*Refine the lattice vectors by finding the 2 that best least squares fit the data
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**latt_vect: std::vector<cv::Vec2i> &, Original estimate of the lattice vectors
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**range: float, Lattice vectors are varied over +/- this range
	**step: float, Step to incrementally move across the component ranges with
	**Returns:
	**std::vector<cv::Vec2f> &, Refined estimate of the lattice vectors
	*/
	std::vector<cv::Vec2f> refine_lattice_vectors(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &latt_vect,	
		int cols, int rows, float range, float step)
	{
		//Express the positions as multiples of the lattice vectors
		std::vector<cv::Point2i> pos = get_pos_as_mult_latt_vect(positions, latt_vect, cols, rows);

		//Calculate limits to iterate between that don't cause the components of the lattice vectors to change sign
		float low1_x = latt_vect[0][0] * (latt_vect[0][0]-range) > 0 ? (latt_vect[0][0]-range) : 0;
		float high1_x = latt_vect[0][0] * (latt_vect[0][0]+range) > 0 ? (latt_vect[0][0]+range) : 0;
		float low2_x = latt_vect[1][0] * (latt_vect[1][0]-range) > 0 ? (latt_vect[1][0]-range) : 0;
		float high2_x = latt_vect[1][0] * (latt_vect[1][0]+range) > 0 ? (latt_vect[1][0]+range) : 0;
		float low1_y = latt_vect[0][1] * (latt_vect[0][1]-range) > 0 ? (latt_vect[0][1]-range) : 0;
		float high1_y = latt_vect[0][1] * (latt_vect[0][1]+range) > 0 ? (latt_vect[0][1]+range) : 0;
		float low2_y = latt_vect[1][1] * (latt_vect[1][1]-range) > 0 ? (latt_vect[1][1]-range) : 0;
		float high2_y = latt_vect[1][1] * (latt_vect[1][1]+range) > 0 ? (latt_vect[1][1]+range) : 0;



		//Interage over the ranges of the ranges of the 4 lattice vector components to find the one that minimises the sum of
		//squared differences from the lattice vector positions
		float min_sqr_diff = FLT_MAX;
		float i_min, j_min, k_min, l_min;
		for (float i = low1_x; i <= high1_x; i += step)
		{
			for (float k = low2_x; k <= high2_x; k += step)
			{
				for (float j = low1_y; j <= high1_y; j += step)
				{
					for (float l = low2_y; l <= high2_y; l += step)
					{
						//Calculate the sum of squared differences from each of the true lattice points
						float sqr_diff = 0;
						for (int m = 1; m < positions.size(); m++)
						{
							sqr_diff += (positions[m].x - pos[m].x*i - pos[m].y*j) * (positions[m].x - pos[m].x*i - pos[m].y*j) + 
								(positions[m].y - pos[m].y*k - pos[m].y*l) * (positions[m].y - pos[m].y*k - pos[m].y*l);

							//Check if this is smaller than the minimum square difference
							if (sqr_diff < min_sqr_diff)
							{
								//Record the parameters that produced this minimum
								min_sqr_diff = sqr_diff;
								i_min = i;
								j_min = j;
								k_min = k;
								l_min = l;
							}
						}
					}
				}
			}
		}

		std::vector<cv::Vec2f> lattice_vectors(2);
		lattice_vectors[0][0] = i_min;
		lattice_vectors[0][1] = j_min;
		lattice_vectors[1][0] = k_min;
		lattice_vectors[1][1] = l_min;

		return lattice_vectors;
	}

	/*Get the spot positions as a multiple of the lattice vectors
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**latt_vect: std::vector<cv::Vec2i> &, Lattice vectors
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**Returns:
	**std::vector<cv::Point2i> &, Spot positions as the nearest integer multiples of the lattice vectors
	*/
	std::vector<cv::Point2i> get_pos_as_mult_latt_vect(std::vector<cv::Point> &positions, std::vector<cv::Vec2i> &latt_vect, 
		int cols, int rows)
	{
		//Assign memory to store the positions as the nearest integer multiple of the lattice positions
		std::vector<cv::Point2i> mult_latt_vect(positions.size());

		//Get all multiples of the lattice vectors that lie in the image
		std::vector<cv::Vec2i> latt_pos = gen_latt_pos(latt_vect, cols, rows, positions[0]);

		//For each position...
		mult_latt_vect[0] = cv::Point2i(0, 0);
        #pragma omp parallel for
		for (int i = 1; i < positions.size(); i++)
		{
			//...find the multiple of the lattice vectors that it is closest to...
			float min_dst2 = INT_MAX;
			int min_dst_idx;
			for (int j = 0; j < latt_pos.size(); j++)
			{
				//...by calculating the squared distances from it
				int dst2 = (positions[i].x - positions[0].x - latt_pos[j][0]) * (positions[i].x - positions[0].x - latt_pos[j][0]) +
					(positions[i].y - positions[0].y - latt_pos[j][1]) * (positions[i].y - positions[0].y - latt_pos[j][1]);

				//Check if this is the minimum distance squared
				if (dst2 < min_dst2)
				{
					min_dst2 = dst2;
					min_dst_idx = j;
				}

				//Record the details of the lattice point of minimum distance
				mult_latt_vect[i] = cv::Point2i(latt_pos[min_dst_idx][0], latt_pos[min_dst_idx][1]);
			}
		}

		return mult_latt_vect;
	}

	/*The positions described by some lattice vectors that lie in a finite plane
	**Input:
	**positions: std::vector<cv::Point>, Positions of located spots. Outlier positions have been removed
	**lattice_vectors: std::vector<cv::Vec2i> &, Lattice vectors
	**cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**origin: cv::Point &, Origin of the lattice that linear combinations of the lattice vectors will be measured relative to
	**Returns:
	**std::vector<cv::Vec2i> &, Linear additive combinations of integer multiples of the lattice vectors that lie in a finite plane
	**with a specified origin. Indices are: 0 - multiple of first lattice vector, 1 - multiple of second lattice vector
	*/
	std::vector<cv::Vec2i> gen_latt_pos(std::vector<cv::Vec2i> &lattice_vectors, int cols, int rows, cv::Point &origin)
	{
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
				while (max_vect1*lattice_vectors[0][0] + origin.x < cols && max_vect1*lattice_vectors[0][1] + origin.y < rows &&
					max_vect1*lattice_vectors[0][0] + origin.x >= 0 && max_vect1*lattice_vectors[0][1] + origin.y >= 0)
				{
					max_vect1++;
				}
			}

            #pragma omp section
			{
				//Calculate the minimum multiple of the first lattice vector from the central spot that is in range
				while (min_vect1*lattice_vectors[0][0] + origin.x < cols && min_vect1*lattice_vectors[0][1] + origin.y < rows &&
					min_vect1*lattice_vectors[0][0] + origin.x >= 0 && min_vect1*lattice_vectors[0][1] + origin.y >= 0)
				{
					min_vect1--;
				}
			}

            #pragma omp section
			{
				//Calculate the maximum multiple of the second lattice vector from the central spot that is in range
				while (max_vect2*lattice_vectors[1][0] + origin.x < cols && max_vect2*lattice_vectors[1][1] + origin.y < rows && 
					max_vect2*lattice_vectors[1][0] + origin.x >= 0 && max_vect2*lattice_vectors[1][1] + origin.y >= 0) 
				{
					max_vect2++;
				}
			}

            #pragma omp section
			{
				//Calculate the minimum multiple of the second lattice vector from the central spot that is in range
				while (min_vect2*lattice_vectors[1][0] + origin.x < cols && min_vect2*lattice_vectors[1][1] + origin.y < rows && 
					min_vect2*lattice_vectors[1][0] + origin.x >= 0 && min_vect2*lattice_vectors[1][1] + origin.y >= 0) 
				{
					min_vect2--;
				}
			}
		}

		//Iterate across multiples of the first lattice vector
		std::vector<cv::Vec2i> latt_pos;
		for (int i = min_vect1; i <= max_vect1; i++)
		{
			//Iterate across multiples of the second lattice vector
			for (int j = min_vect2; j <= max_vect2; j++)
			{
				//Calculate the location of this combination of lattice vectors
				int col = i*lattice_vectors[0][0] + j*lattice_vectors[1][0] + origin.x;
				int row = i*lattice_vectors[0][1] + j*lattice_vectors[1][1] + origin.y;

				//If the lattice point is in the image
				if (row >= 0 && row < rows && col >= 0 && col < cols)
				{
					//Store the position
					latt_pos.push_back(cv::Vec2i(row, col));
				}
			}
		}

		return latt_pos;
	}

	/*Discard the outer spots on the Beanland atlas. Defaults to removing those that are not fully on the aligned diffraction pattern
	**Input:
	**pos: std::vector<cv::Point>, Positions of spots
	**cols: const int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**rows: const int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
	**dst: const int, Minimum distance of spots from the boundary for them to not be discarded
	**Returns:
	**std::vector<cv::Point> &, Spots that are at least the minimum distance from the boundary
	*/
	std::vector<cv::Point> discard_outer_spots(std::vector<cv::Point> &pos, const int cols, const int rows, const int dst)
	{
		//Find the number of inner spots
		int num_inner = 0;
		std::vector<bool> mark_inner(pos.size()); //Mark the positions of the inner spots
		for (int i = 0; i < pos.size(); i++)
		{
			//If the spot is the minimum distance from the boundary, mark its position
			if (pos[i].x >= dst && pos[i].y >= dst && pos[i].x < cols-dst && pos[i].y < rows-dst)
			{
				mark_inner[i] = true;
				num_inner++;
			}
		}

		//Return only the inner spots
		std::vector<cv::Point> inner_spots(num_inner);
		for (int i = 0, k = 0; i < pos.size(); i++)
		{
			if (mark_inner[i])
			{
				inner_spots[k] = pos[i];
				k++;
			}
		}

		return inner_spots;
	}
}