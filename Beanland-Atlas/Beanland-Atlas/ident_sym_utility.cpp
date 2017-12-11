#include <beanland_atlas.h>

namespace ba
{
	/*Get the shift of a second image relative to the first
	**Inputs:
	**img1: cv::Mat &, One of the images
	**img1: cv::Mat &, The second image
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficients and relative row and column shift of the second 
	**image, in that order
	*/
	std::vector<float> quantify_rel_shift(cv::Mat &img1, cv::Mat &img2)
	{
		//Trim padding from the survey region of interest
		cv::Rect roi1 = biggest_not_black(img1);

		//Trim the padding from the second survey region of interest
		cv::Rect roi2 = biggest_not_black(img2);

		//Resize the rois so that they share the smallest rows and smallers columns
		std::vector<cv::Rect> resized_rois = same_size_rois(roi1, roi2);

		//Gaussian blur the regions of interest
		cv::Mat blur1 = blur_by_size(img1(resized_rois[0]));
		cv::Mat blur2 = blur_by_size(img2(resized_rois[1]));

		//Calculate the sum of squared differences
		cv::Mat sum_sqr_diff = ssd(blur1, blur2);

		//Find the minimum sum of squared differences
		cv::Point minLoc;
		cv::minMaxLoc(sum_sqr_diff, NULL, NULL, &minLoc, NULL);

		//Get the relative positions of the centres of highest symmetry
		std::vector<float> sym_pos(3);
		int pad_cols = (int)(QUANT_SYM_USE_FRAC*blur1.cols);
		int pad_rows = (int)(QUANT_SYM_USE_FRAC*blur2.rows);
		sym_pos[1] = roi2.x - roi1.x + pad_cols - minLoc.x;
		sym_pos[2] = roi2.y - roi1.y + pad_rows - minLoc.y;

		//Calculate Pearson normalised product moment correlation coefficient at lowest sum of squared differences and around it
		int search_x = (int)(QUANT_GAUSS_FRAC*blur1.cols);
		int search_y = (int)(QUANT_GAUSS_FRAC*blur1.rows);

		//Calculate range of offsets around minimum sum of squared differences to calculate Pearson normalised product moment 
		//correlation coefficient at
		int max_col_idx = std::min(img1.cols, img2.cols) - 1 - search_x;
		int max_row_idx = std::min(img1.rows, img2.rows) - 1 - search_y;
		int llimx = sym_pos[1] - search_x > -max_col_idx ? sym_pos[1] - search_x : -max_col_idx;
		int ulimx = sym_pos[1] + search_x < max_col_idx ? sym_pos[1] + search_x : max_col_idx;
		int llimy = sym_pos[2] - search_y > -max_row_idx ? sym_pos[2] - search_y : -max_row_idx;
		int ulimy = sym_pos[2] + search_y < max_row_idx ? sym_pos[2] + search_y : max_row_idx;

		//Find the maximum Pearson normalised product moment correlation coefficient in the search range
		float max_pear = -1.0f;
		for (int i = llimx; i <= ulimx; i++)
		{
			for (int j = llimy; j <= ulimy; j++)
			{
				//Check if this coefficient is greater than the maximum
				float pear = pearson_corr(img1, img2, i, j);
				max_pear = pear > max_pear ? pear : max_pear;
			}
		}
		
		//Record the maximum Pearson normalised product moment correlation coefficient
		sym_pos[0] = max_pear;

		return sym_pos;
	}
}