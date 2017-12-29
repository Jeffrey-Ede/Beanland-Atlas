#include <ident_sym_utility.h>

namespace ba
{
	/*Get the shift of a second image relative to the first
	**Inputs:
	**img1: cv::Mat &, One of the images
	**img1: cv::Mat &, The second image
	**use_frac: const float, Fraction of extracted rectangular region of interest to use when calculating the sum of squared 
	**differences to match it against another region
	**wisdom: Information about the images being compared to constrain the relative shift calculation
	**grad_sym_use_frac: const float &, Threshold this portion of the gradient based symmetry values to constrain the regions of 
	**the sum of squared differences when calculating the relative shift
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficient and relative row and column shift of the second 
	**image, in that order
	*/
	std::vector<float> quantify_rel_shift(cv::Mat &img1, cv::Mat &img2, const float use_frac, const int wisdom, 
		const float grad_sym_use_frac)
	{
		//Trim padding from the survey region of interest
		cv::Rect roi1 = biggest_not_black(img1);

		//Trim the padding from the second survey region of interest
		cv::Rect roi2 = biggest_not_black(img2);

		//Resize the rois so that they share the smallest rows and smallers columns
		std::vector<cv::Rect> resized_rois = same_size_rois(roi1, roi2);

		//Gaussian blur the regions of interest, dividing by the number of elements so that sum of squared differences values
		//won't go too high
		cv::Mat blur1 = blur_by_size(img1(resized_rois[0])) / (resized_rois[0].width * resized_rois[0].height);
		cv::Mat blur2 = blur_by_size(img2(resized_rois[1])) / (resized_rois[1].width * resized_rois[1].height);

		//Find the minimum sum of squared differences...
		cv::Point minLoc;

		//...applying wisdom to find the correct relative shift
		if (wisdom == REL_SHIFT_WIS_NONE)
		{
			//Calculate the sum of squared differences
			cv::Mat sum_sqr_diff = ssd(blur1, blur2);

			//Find the minimum sum of squared differences location
			cv::minMaxLoc(sum_sqr_diff, NULL, NULL, &minLoc, NULL);
		}
		else
		{
			if (wisdom == REL_SHIFT_WIS_INTERNAL_MIR0)
			{
				//Calculate the sum of squared differences
				cv::Mat sum_sqr_diff = ssd(blur1, blur2);

				//Calculate the minimum sum of squared differences that is allowed by the mask
				float min_ssd = FLT_MAX;
				for (int i = 0; i < sum_sqr_diff.rows; i++) //Iterate over rows...
				{
					int j = 0; //This mirror symmetry will only shift the rows

					//Check if this is smaller than the minimum
					if (sum_sqr_diff.at<float>(i, j) < min_ssd)
					{
						min_ssd = sum_sqr_diff.at<float>(i, j);
						minLoc.x = j;
						minLoc.y = i;
					}
				}
			}
			else
			{
				if (wisdom == REL_SHIFT_WIS_INTERNAL_MIR1)
				{
					//Calculate the sum of squared differences
					cv::Mat sum_sqr_diff = ssd(blur1, blur2);

					//Calculate the minimum sum of squared differences that is allowed by the mask
					float min_ssd = FLT_MAX;
					float *s;
					int i = 0; //This mirror symmetry will only shift the rows	
					s = sum_sqr_diff.ptr<float>(i);
					for (int j = 0; j < sum_sqr_diff.cols; j++) //...and iterate over columns
					{
						//Check if this is smaller than the minimum
						if (s[j] < min_ssd)
						{
							min_ssd = s[j];
							minLoc.x = j;
							minLoc.y = i;
						}
					}
				}
				else
				{
					//Calculate the minimum ssd allowed by the mask
					if (wisdom == REL_SHIFT_WIS_INTERNAL_ROT)
					{
						//Calculate the sum of squared differences
						cv::Mat sum_sqr_diff = ssd(blur1, blur2);

						//Estimate the symmetry
						cv::Mat est_sym = est_global_sym(blur1, wisdom);

						//Get maximum value
						double max;
						cv::minMaxLoc(est_sym, NULL, &max, NULL, NULL);

						//Calculate the image histogram
						cv::Mat hist;
						int hist_size = GRAD_SYM_HIST_SIZE;
						float range[] = { 0, max };
						const float *ranges[] = { range };
						cv::calcHist(&est_sym, 1, 0, cv::Mat(), hist, 1, &hist_size, ranges, true, false);

						//Work from the top of the histogram to calculate the threshold to use
						float thresh_val;
						const int use_num = grad_sym_use_frac * blur1.rows * blur1.cols;
						for (int i = hist_size-1, tot = 0; i >= 0; i--)
						{
							//Accumulate the histogram bins
							tot += hist.at<float>(i, 1);

							//If the desired total is exceeded, record the threshold
							if (tot > use_num)
							{
								thresh_val = i * max / hist_size;
								break;
							}
						}

						//Threshold the estimated symmetry center values
						cv::Mat thresh;
						cv::threshold(est_sym, thresh, thresh_val, 1, cv::THRESH_BINARY_INV);

						//Calculate the minimum sum of squared differences that is allowed by the mask
						float min_ssd = FLT_MAX;
						float *s;
						for (int i = 0; i < sum_sqr_diff.rows; i++) //Iterate over rows...
						{
							s = sum_sqr_diff.ptr<float>(i);
							for (int j = 0; j < sum_sqr_diff.cols; j++) //...and iterate over columns
							{
								//Check if this is smaller than the minimum
								if (s[j] < min_ssd)
								{
									//Check that this position is allowed by the mask
									if (thresh.at<float>((int)((blur1.rows - i - 1) / 2), (int)((blur1.cols - j - 1) / 2)))
									{
										min_ssd = s[j];
										minLoc.x = j;
										minLoc.y = i;
									}
								}
							}
						}
					}
				}
			}
		}
		
		//Get the relative positions of the centres of highest symmetry
		std::vector<float> sym_pos(3);
		int pad_cols = (int)(use_frac*blur1.cols);
		int pad_rows = (int)(use_frac*blur2.rows);
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
		for (int j = llimx; j <= ulimx; j++)
		{
			for (int i = llimy; i <= ulimy; i++)
			{
				//Calculate Pearson's normalised product moment correlation coefficient
				float pear = pearson_corr(img1, img2, j, i);

				//Check if it is the highest
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

		display_CV(img(cv::Rect(max_row, max_col, max_rows, max_cols)));

		//return cv::Rect(max_col, max_row, max_cols, max_rows);
		return cv::Rect(max_row, max_col, max_rows, max_cols);
	}

	/*Decrease the size of the larger rectangular region of interest so that it is the same size as the smaller
	**Inputs:
	**roi1: cv::Rect, One of the regions of interest
	**roi2: cv::Rect, The other region of interest
	**Returns:
	**std::vector<cv::Rect>, Regions of interest that are the same size, in the same order as the input arguments
	*/
	std::vector<cv::Rect> same_size_rois(cv::Rect roi1, cv::Rect roi2)
	{
		std::vector<cv::Rect> same_size_rects(2);

		//Match up the region of interest rowspans so they are both the smallest
		if (roi1.width > roi2.width)
		{
			roi1.width -= roi1.width - roi2.width;
			roi1.x += (roi1.width - roi2.width) / 2;
		}
		else
		{
			if (roi1.width < roi2.width)
			{
				roi2.width -= roi2.width - roi1.width;
				roi2.x += (roi2.width - roi1.width) / 2;
			}
		}

		//Match up the region of interest columnspans so they are are both the smallest
		if (roi1.height > roi2.height)
		{
			roi1.height -= roi1.height - roi2.height;
			roi1.y += (roi1.height - roi2.height) / 2;
		}
		else
		{
			if (roi1.height < roi2.height)
			{
				roi2.height -= roi2.height - roi1.height;
				roi2.y += (roi2.height - roi1.height) / 2;
			}
		}

		same_size_rects[0] = roi1;
		same_size_rects[1] = roi2;

		return same_size_rects;
	}

	/*Rotates an image keeping the image the same size, embedded in a larger black rectangle
	**Inputs:
	**src: cv::Mat &, Image to rotate
	**angle: float, Angle to rotate the image (anticlockwise) in rad
	**Returns:
	**cv::Mat, Rotated image
	*/
	cv::Mat rotate_CV(cv::Mat src, float angle)
	{
		//Center of rotation
		cv::Point2f pt(src.cols/2., src.rows/2.);

		//Rotation matrix
		cv::Mat rot = getRotationMatrix2D(pt, angle, 1.0);

		//Determine bounding rectangle
		cv::Rect bbox = cv::RotatedRect(pt, src.size(), angle).boundingRect();

		//Adjust the transformation matrix
		rot.at<double>(0,2) += bbox.width/2.0 - pt.x;
		rot.at<double>(1,2) += bbox.height/2.0 - pt.y;

		//Rotate the image
		cv::Mat dst;
		cv::warpAffine(src, dst, rot, bbox.size());

		display_CV(src);
		display_CV(dst);

		return dst;
	}

	/*Order indices in order of increasing angle from the horizontal
	**Inputs:
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices_orig: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	*/
	void order_indices_by_angle(std::vector<float> &angles, std::vector<int> &indices)
	{
		//Structure to combine the angles and indices vectors so that they can be coupled for sorting
		struct angl_idx {
			float angle;
			int idx;
		};

		//Custom comparison of coupled elements based on angle
		struct by_angle { 
			bool operator()(angl_idx const &a, angl_idx const &b) { 
				return a.angle < b.angle;
			}
		};

		//Populate the coupled structure
		std::vector<angl_idx> angls_idxs(angles.size());
		for (int i = 0; i < angles.size(); i++)
		{
			angls_idxs[i].angle = angles[i];
			angls_idxs[i].idx = indices[i];
		}

		//Use the custom comparison to sort the coupled elements
		std::sort(angls_idxs.begin(), angls_idxs.end(), by_angle());

		//Replace the contents of the original vectors with the ordered vlues
		for (int i = 0; i < angles.size(); i++)
		{
			angles[i] = angls_idxs[i].angle ;
			indices[i] = angls_idxs[i].idx;
		}
	}

	/*Estimate the global symmetry center of an image by finding the pixel that has the closest to zero total Scharr filtrate
	**when it it summed over equidistances in all directions from that point
	**Note: this function's speed can be increased by saving sums rather than repeatedly recalculating them
	**Inputs:
	**img: cv::Mat &, Floating point greyscale OpenCV mat to find the global symmetry center of
	**wisdom: const int, Information about the symmetry that allows a gradients to be summed over a larger area
	**not_calc_val: const float, Value to set elements that do not have at least the minimum area to perform their calculation
	**use_frac: const float, Only global symmetry centers at least this fraction of the image's area will be considered
	**edges will be considered
	**Returns:
	**cv::Mat, Sums of gradients in the largest possible rectangular regions centred on each pixel
	*/
	cv::Mat est_global_sym(cv::Mat &img, const int wisdom, const float not_calc_val, const float use_frac)
	{
		//Aquire the image's Sobel filtrate
		cv::Mat gradx, grady;
		cv::Sobel(img, gradx, CV_32FC1, 0, 1, CV_SCHARR, 1.0f / (img.rows*img.cols)); //Division helps safeguard against overflow
		cv::Sobel(img, grady, CV_32FC1, 1, 0, CV_SCHARR, 1.0f / (img.rows*img.cols)); //Division helps safeguard against overflow

		//Calculate the minimum distance from the edge
		int min_area = (int)(use_frac * img.rows*img.cols);
		int min_cols = min_area / img.rows;
		int min_rows = min_area / img.cols;

		//Initialise the global symmetry weightings matrix to store the ouput of the calculations
		cv::Mat sym = cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(not_calc_val));

		if (wisdom == REL_SHIFT_WIS_NONE || wisdom == REL_SHIFT_WIS_INTERNAL_ROT)
		{
			//Find the global symmetry values in the left half of the image
			for (int i = min_cols; i < img.cols-min_cols; i++)
			{
				//Number of columns to sum over
				int num_cols = std::min(i, img.cols-i-1);

				for (int j = min_rows; j < img.rows-min_rows; j++)
				{
					//Number of rows to sum over
					int num_rows = std::min(j, img.rows-j-1);

					//Check that the area that gradient contributions will be summed over is greater than the minimum area
					if ((2 * num_cols + 1) * (2 * num_rows + 1) >= min_area)
					{
						//Sum the gradients in all directions from the point, but not at the point
						float sumx = -gradx.at<float>(j, i), sumy = -grady.at<float>(j, i);

						//Sum gradients in symmetric region centered on the point of interest
						for (int k = j-num_rows; k <= j+num_rows; k++)
						{
							for (int l = i-num_cols; l <= i+num_cols; l++)
							{
								sumx += gradx.at<float>(k, l);
								sumy += grady.at<float>(k, l);
							}
						}

						sym.at<float>(j, i) = std::abs(sumx) + std::abs(sumy);
					}			
				}
			}
		}
		else
		{
			if (wisdom == REL_SHIFT_WIS_INTERNAL_MIR0)
			{
				std::vector<float> sum_rows(img.rows, 0); //Sum of all the y gradient elements in a row
				float *p;
				//Iterate over rows...
				for (int i = 0; i < img.rows; i++)
				{
					//...and iterate over columns
					p = grady.ptr<float>(i);
					for (int j = 0; j < img.cols; j++)
					{
						sum_rows[i] += p[j];
					}
				}

				//Find the global symmetry values in the left half of the image
				for (int i = min_cols; i < img.cols-min_cols; i++)
				{
					//Number of columns to sum over
					int num_cols = std::min(i, img.cols-i-1);

					for (int j = min_rows; j < img.rows-min_rows; j++)
					{
						//Number of rows to sum over
						int num_rows = std::min(j, img.rows-j-1);

						//Check that the area that gradient contributions will be summed over is greater than the minimum area
						if ((2 * num_cols + 1) * (2 * num_rows + 1) >= min_area)
						{
							//Sum the gradients in all directions from the point, but not at the point
							float sumx = -gradx.at<float>(j, i), sumy = -grady.at<float>(j, i);

							//Sum gradients in symmetric region centered on the point of interest
							for (int k = j-num_rows; k <= j+num_rows; k++)
							{
								for (int l = i-num_cols; l <= i+num_cols; l++)
								{
									sumx += gradx.at<float>(k, l);
								}

								//All y gradient columns are summed across as the symmetry is known
								sumy += sum_rows[k];
							}

							sym.at<float>(j, i) = std::abs(sumx) + std::abs(sumy);
						}			
					}
				}
			}
			else
			{
				if (wisdom == REL_SHIFT_WIS_INTERNAL_MIR1)
				{
					std::vector<float> sum_cols(img.cols, 0); //Sum of all the y gradient elements in a row
					//Iterate over columns...
					for (int i = 0; i < img.cols; i++)
					{
						//...and iterate over rows
						for (int j = 0; j < img.rows; j++)
						{
							sum_cols[i] += gradx.at<float>(j, i);
						}
					}

					//Find the global symmetry values in the left half of the image
					for (int i = min_cols; i < img.cols-min_cols; i++)
					{
						//Number of columns to sum over
						int num_cols = std::min(i, img.cols-i-1);

						for (int j = min_rows; j < img.rows-min_rows; j++)
						{
							//Number of rows to sum over
							int num_rows = std::min(j, img.rows-j-1);

							//Check that the area that gradient contributions will be summed over is greater than the minimum area
							if ((2 * num_cols + 1) * (2 * num_rows + 1) >= min_area)
							{
								//Sum the gradients in all directions from the point, but not at the point
								float sumx = -gradx.at<float>(j, i), sumy = -grady.at<float>(j, i);

								//Sum gradients in symmetric region centered on the point of interest
								for (int l = i-num_cols; l <= i+num_cols; l++)
								{
									for (int k = j-num_rows; k <= j+num_rows; k++)
									{
										sumy += grady.at<float>(k, l);
									}

									//All x gradient columns are summed across as the symmetry is known
									sumx += sum_cols[l];
								}

								sym.at<float>(j, i) = std::abs(sumx) + std::abs(sumy);
							}			
						}
					}
				}
			}
		}

		return sym;
	}
}