#include <beanland_atlas.h>

namespace ba
{
	/*Downsamples amalgamation of aligned diffraction patterns, then finds approximate axes of symmetry
	**Inputs:
	**amalg: cv::Mat &, OpenCV mat containing a diffraction pattern to find the axes of symmetry of
	**origin_x: int, Position of origin accross the OpenCV mat
	**origin_y: int, Position of origin down the OpenCV mat
	**num_angles: int, Number of angles  to look for symmetry at
	**target_size: int, Downsampling factor will be the largest power of 2 that doesn't make the image smaller than this
	**Returns:
	**std::vector<float>, highest Pearson normalised product moment correlation coefficient for a symmetry lins drawn through 
	**a known axis of symmetry
	*/
	std::vector<float> symmetry_axes(cv::Mat &amalg, int origin_x, int origin_y, size_t num_angles, float target_size) {

		float inv_num_angles = 1.0f/num_angles;

		float offset_accr = (float)origin_x;
		float offset_down = (float)origin_y;

		cv::Mat d_samp;
		if(target_size && target_size != std::min(amalg.rows, amalg.cols)){

			float d_scale = target_size / std::min(amalg.rows, amalg.cols);
			cv::resize(amalg, d_samp, cv::Size(), d_scale, d_scale, cv::INTER_LANCZOS4);

			offset_accr *= d_scale;
			offset_down *= d_scale;
		}
		else {
			d_samp = amalg;
		}

		std::vector<float> pearson_corr(num_angles);

		//For each angle
		#pragma omp parallel for 
		for (int k = 0; k < num_angles; k++){

			//Gradients for reflection
			float grad_accr = std::cos(k*PI*inv_num_angles);
			float grad_down = std::sin(k*PI*inv_num_angles);
			float inv_grad2 = 1.0f/(grad_accr*grad_accr+grad_down*grad_down);

			//Sums for Pearson product moment correlation coefficient
			int count = 0;
			float sum_xy = 0.0f;
			float sum_x = 0.0f;
			float sum_y = 0.0f;
			float sum_x2 = 0.0f;
			float sum_y2 = 0.0f;

			//Reflect points across mirror line and compare them to the nearest pixel
			for (int i = 0; i < d_samp.cols; i++) {
				for (int j = 0; j < d_samp.rows; j++) {

					//Get coordinates relative to center of symmetry
					int x = i-offset_accr;
					int y = j-offset_down;

					//Only perform calculation on points on one side of the mirror line
					if(x*grad_down-y*grad_accr > 0.0f){

						//Reflect
						int refl_col = 2*(int)((grad_accr*x+grad_down*y)*inv_grad2)*grad_accr - x + offset_accr;
						int refl_row = 2*(int)((grad_accr*x+grad_down*y)*inv_grad2)*grad_down - y + offset_down;

						//If reflected point is in the image, add it's contribution to Pearson's correlation coefficient
						if(refl_col < d_samp.cols && refl_row < d_samp.rows && refl_col > 0 && refl_row > 0){

							//Contribute to Pearson correlation coefficient
							count++;
							sum_xy += d_samp.at<float>(j, i)*d_samp.at<float>(refl_row, refl_col);
							sum_x += d_samp.at<float>(j, i);
							sum_y += d_samp.at<float>(refl_row, refl_col);
							sum_x2 += d_samp.at<float>(j, i)*d_samp.at<float>(j, i);
							sum_y2 += d_samp.at<float>(refl_row, refl_col)*d_samp.at<float>(refl_row, refl_col);
						}
					}
				}
			}

			pearson_corr[k] = (count*sum_xy - sum_x*sum_y) / (std::sqrt(count*sum_x2 - sum_x*sum_x) * std::sqrt(count*sum_y2 - sum_y*sum_y));
		}

		return pearson_corr;
	}
}