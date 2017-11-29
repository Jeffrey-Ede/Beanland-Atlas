#include <beanland_atlas.h>

/**Refine positions of mirror lines. Use bivariate optimisation to get best centre to subpixel accuracy
**Inputs:
**amalg: cv::Mat, Diffraction pattern to refine symmetry lines on
**max_pos: std::vector<int>, Array indices corresponding to intensity maxima
**num_angles: max_pos indices are converted to angles via angle_i = max_pos[i]*PI/num_angles
**Returns:
**std::vector<cv::Vec3f>, Refined origin position and angle of each mirror line, in the same order as the input maxima
*/
std::vector<cv::Vec3f> refine_mir_pos(cv::Mat amalg, std::vector<int> max_pos, size_t num_angles, int origin_x, int origin_y, int range)
{

	//Vector to hold refined mirror lines
	std::vector<cv::Vec3f> mirror_lines(max_pos.size());

	//Factor to scale max_pos indices by to get the angle they correspond to
	float idx_to_rad = PI/(float)num_angles;

	//Refinement sweep resolution will be 2pi/(NUM_PERIM*micrograph perimeter)
	float sweep_res = PI/(NUM_PERIM*(amalg.rows+amalg.cols));

	//Number of angles to sweep through during refinement
	int num_sweep_angles = (int)std::ceil(2*idx_to_rad/sweep_res);

	//Number of positions around calculated centre of symmetry to calculate Pearson product moment correlation coefficients for
	int stride = 2*range+1;
	int num_test_pos = stride*stride;

	//Refine position of each mirror line
	for (int m = 0; m < max_pos.size(); m++) {

		//Explore Pearson product moment correlation coefficient space until maxima is found
		float expl_accr = 0, expl_down = 0; //Offset from origin
		bool explore = true;
		std::vector<float> explored_loc(2*MAX_EXPL); //Save previous positions
		int num_expl = 0; //Number of regions of Pearson coefficient space mapped out

		while (explore){

			num_expl++;

			//Vector to hold highest Pearson normalised product moment coefficient for each test position
			std::vector<float> coefficients(num_test_pos);
			std::vector<float> angles(num_test_pos);

			//For each pixel in range of approximate center of symmetry, going across
			for (int incr_accr = -range, accr_count = 0; incr_accr <= range; incr_accr++, accr_count++) {

				//Check that the row is in the micrograph. It is very unlikely it is not, but check anyway
				if (origin_x+incr_accr+expl_accr < amalg.cols) {
				
					//Going down
					for (int incr_down = -range, down_count = 0; incr_down <= range; incr_down++, down_count++) {

						//Check that the column is in the micrograph. It is very unlikely it is not, but check anyway
						if(origin_y+incr_down+expl_down < amalg.rows){

							//Sweep within +/- the angular resolution of the approximate mirror line location method
							// and calculate Pearson's product moment correlation coefficient for each mirror line
							float starting_angle = max_pos[m] * idx_to_rad - idx_to_rad;
							std::vector<float> pearson_corr(num_sweep_angles);
		
							{
								float angle; int sweep_count; //Loop variables are explicitely contained in their own lexical scope
								for (sweep_count = 0, angle = starting_angle; angle < max_pos[m] * idx_to_rad + idx_to_rad;
									angle += sweep_res, sweep_count++) {

									//Gradients for reflection
									float grad_accr = std::cos(angle);
									float grad_down = std::sin(angle);
									float inv_grad2 = 1.0f/(grad_accr*grad_accr+grad_down*grad_down);

									//Sums for Pearson product moment correlation coefficient
									int count = 0;
									float sum_xy = 0.0f;
									float sum_x = 0.0f;
									float sum_y = 0.0f;
									float sum_x2 = 0.0f;
									float sum_y2 = 0.0f;

									//Reflect points across mirror line and compare them to the nearest pixel
                                    #pragma omp parallel for
									for (int i = 0; i < amalg.cols; i++) {
										for (int j = 0; j < amalg.rows; j++) {

											//Get coordinates relative to center of symmetry
											int x = i-origin_x-incr_accr-expl_accr;
											int y = j-origin_y-incr_down-expl_down;

											//Only perform calculation on points on one side of the mirror line
											if(x*grad_down-y*grad_accr > 0.0f){

												//Reflect
												int refl_col = 2*(int)((grad_accr*x+grad_down*y)*inv_grad2)*grad_accr - x +
													origin_x+incr_accr+expl_accr;
												int refl_row = 2*(int)((grad_accr*x+grad_down*y)*inv_grad2)*grad_down - y +
													origin_y+incr_down+expl_down;

												//If reflected point is in the image, add it's contribution to Pearson's correlation coefficient
												if(refl_col < amalg.cols && refl_row < amalg.rows && refl_col > 0 && refl_row > 0){

													//Contribute to Pearson correlation coefficient
													count++;
													sum_xy += amalg.at<float>(j, i)*amalg.at<float>(refl_row, refl_col);
													sum_x += amalg.at<float>(j, i);
													sum_y += amalg.at<float>(refl_row, refl_col);
													sum_x2 += amalg.at<float>(j, i)*amalg.at<float>(j, i);
													sum_y2 += amalg.at<float>(refl_row, refl_col)*amalg.at<float>(refl_row, refl_col);
												}
											}
										}
									}

									//Record Pearson product moment correlation coefficient
									pearson_corr[sweep_count] = (count*sum_xy - sum_x*sum_y) / (std::sqrt(count*sum_x2 - sum_x*sum_x) *
										std::sqrt(count*sum_y2 - sum_y*sum_y));
								}
							}

							//Record the value of the maximum coefficient and the angle it corresponds to
							int max_corr_idx = std::distance(pearson_corr.begin(), std::max_element(pearson_corr.begin(), pearson_corr.end()));
							coefficients[down_count*stride+accr_count] = pearson_corr[max_corr_idx];
							angles[down_count*stride+accr_count] = starting_angle+max_corr_idx*sweep_res;
						}
						else {
							continue;
						}
					}
				}
				else {
					continue;
				}
			}

			//Find the position of the maximum
			int max_idx = std::distance(coefficients.begin(), std::max_element(coefficients.begin(), coefficients.end()));

			//Keep exploring while the maxima is an edge pixel
			if (max_idx < stride || max_idx >= stride*(stride - 1) || max_idx % stride == 0 || max_idx % stride == stride - 1) {

				explored_loc[2*num_expl-2] = expl_accr;
				explored_loc[2*num_expl-1] = expl_down;

				/* Estimate the position of the maxima using the region's 1st, 2nd and 3rd derivatives */

				//First derivative down
				float d_1_down = -coefficients[0];
				for (int i = 1; i < stride; i++) {
					d_1_down -= coefficients[i];
				}
				
				for (int i = num_test_pos - stride; i < num_test_pos; i++) {
					d_1_down += coefficients[i];
				}
				d_1_down /= 2*stride;

				//First derivative accross
				float d_1_accr = -coefficients[0];
				for (int i = stride; i < num_test_pos; i += stride) {
					d_1_accr -= coefficients[i];
				}

				for (int i = stride-1; i < num_test_pos; i += stride) {
					d_1_accr += coefficients[i];
				}
				d_1_accr /= 2*stride;

				//Second derivative down
				float d_2_down = +coefficients[0];
				for (int i = 1; i < stride; i++) {
					d_2_down += coefficients[i];
				}

				for (int i = stride/2; i < num_test_pos; i += stride) {
					d_2_down -= 2*coefficients[i];
				}

				for (int i = num_test_pos - stride; i < num_test_pos; i+= stride) {
					d_2_down += coefficients[i];
				}
				d_2_down /= range;

				//Second derivative across
				float d_2_accr = +coefficients[0];
				for (int i = 1; i < stride; i++) {
					d_2_accr += coefficients[i];
				}

				for (int i = stride/2; i < num_test_pos; i += stride) {
					d_2_accr -= 2*coefficients[i];
				}

				for (int i = num_test_pos - stride; i < num_test_pos; i++) {
					d_2_accr += coefficients[i];
				}
				d_2_accr /= range*range;

				/* Perform checks before extrapolating maxima position */
				//If 2nd derivative is very small, don't use it to estimate maxima position
				if (std::abs(d_2_down) < MIN_D_2) {
					if (d_1_down > 0) {
						expl_down += stride;
					}
					else {
						expl_down -= stride;
					}
				}
				else {
					//Use derivatives to estimate the position of the maximum if 2nd derivative is sufficiently large
					if (d_1_down*d_2_down > 0) {
						expl_down -= std::max((float)(0.5*d_1_down/d_2_down), (float)stride);
					}
					else {
						expl_down += stride*std::copysignf(1.0, d_1_down);
					}
				}

				//If 2nd derivative is very small, don't use it to estimate maxima position
				if (std::abs(d_2_accr) < MIN_D_2) {
					if (d_1_down > 0) {
						expl_accr += stride;
					}
					else {
						expl_accr -= stride;
					}
				}
				else{
					//Use derivatives to estimate the position of the maximum if 2nd derivative is sufficiently large
					if (d_1_down*d_2_down > 0) {
						expl_accr -= std::max((float)(0.5*d_1_accr/d_2_accr), (float)stride);
					}
					else {
						expl_accr += stride*std::copysignf(1.0, d_1_accr);
					}
				}

				//Check to make sure that solution isn't oscillating around the same positions
				if (num_expl > 1) {
					//Comparing next locations against previous
					for (int i = 2*num_expl-1; i > 0; i -= 2) {

						//If next location is within range of a previous position, stop exploring and use this maxima to parameterise
						//the mirror line
						if (std::abs(expl_accr - explored_loc[i-1] <= range) && std::abs(expl_down - explored_loc[i] <= range)) {

							//Get mirror line parameters
							int rem = max_idx%stride;
							int div = max_idx/stride;
							float accr_loc = origin_x + rem-range + expl_accr;
							float down_loc = origin_y + div-range + expl_down;

							mirror_lines[m] = cv::Vec3f(accr_loc, down_loc, angles[max_idx]);

							explore = false;
						}
					}
				}
			}
			//Maxima has been found!
			else {
				
				//Get mirror line parameters
				int rem = max_idx%stride;
				int div = max_idx/stride;
				float accr_loc = origin_x + rem-range + expl_accr;
				float down_loc = origin_y + div-range + expl_down;

				mirror_lines[m] = cv::Vec3f(accr_loc, down_loc, angles[max_idx]);

				explore = false;
			}
		}
	}

	return mirror_lines;
}

/*Calculate the mean estimate for the centre of symmetry
**Input:
**refined_pos: std::vector<cv::Vec3f>, 2 ordinates and the inclination of each line
**Returns:
**cv::Vec2f, The average 2 ordinates of the centre of symmetry
*/
cv::Vec2f avg_origin(std::vector<cv::Vec3f> lines)
{
	const int len = lines.size();

	float x_avg = 0.0f, y_avg = 0.0f;

	//For each unique combination
	for (int i = 0; i < len; i++) {

		x_avg += lines[i][0];
		y_avg += lines[i][1];
	}

	return cv::Vec2f(x_avg/len, y_avg/len);
}

/*Calculate all unique possible points of intersection, add them up and then divide by their number to get the average
**Input:
**refined_pos: std::vector<cv::Vec3f>, 2 ordinates and the inclination of each line
**Returns:
**cv::Vec2f, The average 2 ordinates of intersection
*/
cv::Vec2f average_intersection(std::vector<cv::Vec3f> lines) 
{
	const int len = lines.size();
	
	float x_avg = 0.0f, y_avg = 0.0f;

	//For each unique combination
	for (int i = 0; i < len; i++) {
		for (int j = i+1; j < len; j++) {
			
			float denom = (lines[i][0]-lines[i][0]-std::cos(lines[i][2])) * (lines[j][1]-lines[j][1]-std::sin(lines[j][2])) - 
				(lines[i][1]-lines[i][1]-std::sin(lines[i][2])) * (lines[j][0]-lines[j][0]-std::cos(lines[j][2]));

			float factor_1 = lines[i][0]*(lines[i][1]+std::sin(lines[i][2])) - lines[i][1]*(lines[i][0]+std::cos(lines[i][2]));
			float factor_2 = lines[j][0]*(lines[j][1]+std::sin(lines[j][2])) - lines[j][1]*(lines[j][0]+std::cos(lines[j][2]));

			x_avg += ( factor_2*std::cos(lines[i][2]) - factor_1*std::cos(lines[j][2]) ) / denom;
			y_avg += ( factor_2*std::sin(lines[i][2]) - factor_1*std::sin(lines[j][2]) ) / denom;
		}
	}

	long unsigned int combinations = factorial(len);

	return cv::Vec2f(x_avg/combinations, y_avg/combinations);
}