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

	/*Get the largest rectangular portion of an image inside a surrounding black background
	**Inputs:
	**img: cv::Mat &, Input floating point image to extract the largest possible non-black region of
	**Returns:
	**cv::Rect, Largest non-black region
	*/
	cv::Rect biggest_not_black(cv::Mat &img)
	{
		//Work inwards from the top and bottom of every column to find the length of the non-black region
		std::vector<int> col_thick(img.cols);
		std::vector<int> col_top(img.cols);
		std::vector<int> col_bot(img.cols);
		for (int i = 0; i < img.cols; i++)
		{
			//From the top of the image
			int j_top;
			for (j_top = 0; j_top < img.rows; j_top++)
			{
				if (img.at<float>(j_top, i))
				{
					break;
				}
			}

			//From the bottom of the image
			int j_bot;
			for (j_bot = img.rows-1; j_bot >= 0; j_bot--)
			{
				if (img.at<float>(j_bot, i))
				{
					break;
				}
			}

			col_thick[i] = j_bot > j_top ? j_bot - j_top + 1 : 0;
			col_top[i] = j_top;
			col_bot[i] = j_bot;
		}

		//Work inwards from the top and bottom of every row to find the length of the non-black region
		std::vector<int> row_thick(img.rows);
		std::vector<int> row_top(img.rows);
		std::vector<int> row_bot(img.rows);
		for (int i = 0; i < img.rows; i++)
		{
			//From the top of the image
			int j_top;
			for (j_top = 0; j_top < img.cols; j_top++)
			{
				if (img.at<float>(i, j_top))
				{
					break;
				}
			}

			//From the bottom of the image
			int j_bot;
			for (j_bot = img.cols-1; j_bot >= 0; j_bot--)
			{
				if (img.at<float>(i, j_bot))
				{
					break;
				}
			}

			row_thick[i] = j_bot > j_top ? j_bot - j_top + 1 : 0;
			row_top[i] = j_top;
			row_bot[i] = j_bot;
		}

		//Find the combination that has the highest area
		int max_area = -1;
		int max_row, max_col, max_rows, max_cols;
		for (int i = 0; i < img.rows; i++)
		{
			//For each non-black region of rows with non-zero thickness, find the column of minimum thickness
			if(row_thick[i])
			{
				int min_thick_vert_idx = std::distance(col_thick.begin(),
					std::min_element(col_thick.begin() + row_top[i], col_thick.begin() + row_bot[i]));

				if(col_thick[min_thick_vert_idx])
				{
					//Find the minimum row thickness for the rowspan of this row of minimum thickness
					int min_thick_horiz_idx = std::distance(row_thick.begin(),
						std::min_element(row_thick.begin() + col_top[min_thick_vert_idx], row_thick.begin() + col_bot[min_thick_vert_idx]));

					////Find the differences between the top and bot columns of the rows at the top and bottom of this column of
					////minimum thickness
					//int top_row_thick = row_thick[col_top[min_thick_vert_idx]];
					//int bot_row_thick = row_thick[col_bot[min_thick_vert_idx]];



					//Check if the area is higher than the maximum...
					int area = row_thick[min_thick_horiz_idx]*col_thick[min_thick_vert_idx];
					if (area > max_area)
					{
						//...and update the roi parameters if it is the largest
						max_area = area;
						max_row = row_top[min_thick_horiz_idx];
						max_col = col_top[min_thick_vert_idx];
						max_rows = row_thick[min_thick_horiz_idx];
						max_cols = col_thick[min_thick_vert_idx];
					}
				}
			}
		}

		return cv::Rect(max_row, max_col, max_rows, max_cols);
	}
}