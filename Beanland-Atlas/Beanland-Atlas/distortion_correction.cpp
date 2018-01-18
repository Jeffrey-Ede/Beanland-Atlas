#include <distortion_correction.h>

namespace ba
{
	/*Use spot overlaps to determine the distortion field
	**Inputs:
	**groups: std::vector<cv::Mat> &, Preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	**group_pos: std::vector<cv::Point> &, Positions of top left corners of circles' bounding squares
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**grouped_idx: std::vector<std::vector<int>> &, Groups of consecutive image indices where the spots are all in the same position
	**is_in_img: std::vector<bool> &, True when the spot is in the image so that indices can be grouped
	**radius: const int, Radius of the spots
	**col_max: const int, Maximum column difference between spot positions
	**row_max: const int, Maximum row difference between spot positions
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	*/
	void get_aberrating_fields(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam)
	{
		//Initialise the MATLAB engine
		//matlab::data::ArrayFactory factory;
		//std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::connectMATLAB();

		//Get the spots that are fully in images
		std::vector<std::vector<double>> mirr_lines;
		std::vector<cv::Vec2i> combinations;
		for (int k = 0, m = 0, counter = 0; k < grouped_idx.size(); k++)
		{
			if (is_in_img[k])
			{
				for (int l = k+1, n = m+1; l < grouped_idx.size(); l++)
				{
					if (is_in_img[l])
					{
						//Find the overlapping region between the circles
						circ_overlap co = get_overlap(spot_pos, rel_pos, col_max, row_max, radius, grouped_idx[m][0], grouped_idx[n][0], 
							cols, rows);

						//Check if they overlap and the overlapping region is fully in the image
						if (co.overlap)
						{
							//Find if the overlapping region is entirely on the image
							if (on_img(co.bounding_rect[0], cols, rows) && on_img(co.bounding_rect[1], cols, rows))
							{
								//Create a mask idicating where the circles overlap to extract the values from the images
								cv::Mat co_mask = gen_circ_overlap_mask(co.P1, radius, co.P2, radius, cols, rows);

								//Get the number of overlapping pixels from the mask
								int num_overlap = cv::countNonZero(co_mask);

								//Check that there are enough pixels to make a meaningful estimate of the symmetry center
								if (num_overlap > MIN_OVERLAP_PX_NUM)
								{
									//Parameters describing each overlap. By index: 0 - Fraction of circle radius from the first circle's center,
									//1 - Fraction of circle radius from the second circle's center, 2 - ratio of the first circle's pixel value
									//to the second circle's
									std::vector<cv::Vec3d> overlaps(num_overlap);

									//Get the ratios of the overlapping protions of the image
									cv::Mat ratios, ratios_mask; //Overlapping intensity ratios; smaller divided by larger
									cv::Point pos; //Positions of ratios and ratios_mask in groups[m]
									get_overlap_ratios(co_mask, groups[m], groups[n], group_pos[m], group_pos[n], ratios, ratios_mask, pos);

									////Package data into vectors so that it can be packed for MATLAB by the ArrayFactory
									//float p; //Indicates the OpenCV mat type to the templated function
									//std::vector<double> ratios_vect;
									//cvMat_to_vect( ratios, ratios_vect, p );
									//
									//std::vector<double> ratios_mask_vect;
									//cvMat_to_vect( ratios_mask, ratios_mask_vect, p );

									////Estimated axis of symmetry.
									//std::vector<double> line1(4);
									//line1[0] = co.maxima[0].x;
									//line1[1] = co.maxima[0].y;
									//line1[2] = co.maxima[1].x;
									//line1[3] = co.maxima[1].y;

									//std::vector<double> line2(4);
									//line2[0] = co.minima[0].x;
									//line2[1] = co.minima[0].y;
									//line2[2] = co.minima[1].x;
									//line2[3] = co.minima[1].y;

									////Distance between estimated circle centers
									//double dist = std::sqrt( (co.P1.x-co.P2.x)*(co.P1.x-co.P2.x) + (co.P1.y-co.P2.y)*(co.P1.y-co.P2.y) );

									////Maximum magnitudes of translations and rotations that can be trialed
									//std::vector<double> region(3);
									//region[0] = DISTORT_MAX_REL_POS_ERR;
									//region[1] = DISTORT_MAX_REL_POS_ERR;
									//region[2] = std::atan( SQRT_OF_2 * DISTORT_MAX_REL_POS_ERR / dist );

									//matlab::data::ArrayFactory factory;
									//std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::connectMATLAB();

									////Package data for the MATLAB function
									//std::vector<matlab::data::Array> args({
									//	factory.createArray( { (size_t)ratios.rows, (size_t)ratios.cols }, ratios_vect.begin(), ratios_vect.end() ),
									//	factory.createArray( { (size_t)ratios_mask.rows, (size_t)ratios_mask.cols },
									//	    ratios_mask_vect.begin(), ratios_mask_vect.end() ),
									//	factory.createArray( { 1 , 4 }, line1.begin(), line1.end() ),
									//	factory.createArray( { 1 , 4 }, line2.begin(), line2.end() ),
									//	factory.createArray( { 1 , 3 }, region.begin(), region.end() ),
									//	factory.createScalar<double>( LS_TOL ),
									//	factory.createScalar<int32_t>( LS_MAX_ITER )
									//});

									////Pass data to MATLAB to calculate the approximately 2mm symmery center
									//matlab::data::TypedArray<double> const mirr_lines_info = matlabPtr->feval(
									//	matlab::engine::convertUTF8StringToUTF16String("get_mirr_sym"), args);

									//std::vector<double> lines(8);
									//{
									//	int k = 0;
									//	for (auto val : mirr_lines_info)
									//	{
									//		lines[k++] = val;
									//	}
									//}

									////Store information about the mirror line combination
									//mirr_lines.push_back(lines);
									//combinations.push_back(cv::Vec2i(m, n));

									//std::getchar();
								}
							}
						}

						//Go to th next group
						n++;
					}
				}

				//Go to next group
				m++;
			}
		}
	}

	/*Get the ratio of the intensity profiles where 2 spots overlap
	**Inputs:
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate where the circles overlap
	**c1: cv::Mat &, One of the circles
	**c2; cv::Mat &, The other circles
	**p1: cv::Point &, Top left point of a square containing the first circle on the detector
	**p2: cv::Point &, Top left point of a square containing the second circle on the detector
	**max_shift: const float, Maximum shift to consider
	**incr_shift: const float, Steps to adjust the shift by
	**ratios: cv::Mat &, Output cropped image containing the overlap region containing the ratios of the intensity profiles.
	**Ratios larger than 1.0 are reciprocated
	**ratios_mask: cv::Mat &, Output mask indicating the ratios pixels containing ratios
	**rel_pos: cv::Point &, The position of the top left corner of ratios and ratios_mask in the c1 mat
	*/
	void get_overlap_ratios(cv::Mat &mask, cv::Mat &c1, cv::Mat &c2, cv::Point &p1, cv::Point &p2, 
		cv::Mat &ratios, cv::Mat &ratios_mask, cv::Point &rel_pos)
	{
		//Create an image to store ratios on and a mask indicating them
		cv::Mat pre_crop_ratios = cv::Mat(c1.size(), CV_32FC1, cv::Scalar(0.0));

		//Get the pixel value and distances from circle centers for pixels in the overlapping region
		byte *b;
		for (int i = 0, co_num = 0; i < mask.rows; i++)
		{
			b = mask.ptr<byte>(i);
			for (int j = 0; j < mask.cols; j++)
			{
				//If the pixel is marked as an overlapping region pixel
				if (b[j])
				{
					//Get the values of the pixels
					double val1, val2;
					val1 = c1.at<float>(i-p1.y, j-p1.x);
					val2 = c2.at<float>(i-p2.y, j-p2.x);

					pre_crop_ratios.at<float>(i-p1.y, j-p1.x) = val1 > val2 ? val2 / val1 : val1 / val2;
				}
			}
		}

		//Get the rectangle needed to crop the images to reduce the amound of memory needed to store them
		cv::Rect rect = get_non_zero_mask_px_bounds(mask);

		//Output the overlap-containing region, along with it's position in the first spot mat
		pre_crop_ratios(rect).copyTo(ratios);
		mask(rect).copyTo(ratios_mask);
		rel_pos = cv::Point(rect.x, rect.y);
	}

	/*Create a rectangle containing the non-zero pixels in a mask that can be used to crop them from it
	**Inputs:
	**mask: cv::Mat &, 8-bit mask image
	**Returns:
	**cv::Rect, Rectangle containing the non-zero pixels in a mask
	*/
	cv::Rect get_non_zero_mask_px_bounds(cv::Mat &mask)
	{
		//Find the minimum and maximum rows and columns
		byte *b;
		int min_row = INT_MAX, max_row = 0, min_col = INT_MAX, max_col = 0;
		for (int i = 0; i < mask.rows; i++)
		{
			b = mask.ptr<byte>(i);
			for (int j = 0; j < mask.cols; j++)
			{
				//Check if the non-zero mask points are extrema
				if (b[j])
				{
					if (i < min_row)
					{
						min_row = i;
					}
					if (i > max_row)
					{
						max_row = i;
					}
					if (j < min_col)
					{
						min_col = j;
					}
					if (j > max_col)
					{
						max_col = j;
					}
				}
			}
		}

		return cv::Rect(min_col, min_row, max_col-min_col+1, max_row-min_row+1);
	}

	/*Calculate the affine transform that best matches the overlapping region between 2 overlapping circles. Not currently
	**being used. May finish this function later
	**co: circ_overlap &, Region where circles overlap
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate where the circles overlap
	**c1: cv::Mat &, One of the circles
	**c2; cv::Mat &, The other circles
	**p1: cv::Point &, Top left point of a square containing the first circle on the detector
	**p2: cv::Point &, Top left point of a square containing the second circle on the detector
	**max_affine_shift: const float, Maximum amount to try shifting affine control points
	**incr_affine_shift: const float, Increment between affine control points being trialled
	*/
	void get_best_overlap(circ_overlap &co, cv::Mat &mask, cv::Mat &c1, cv::Mat &c2, cv::Point &p1, cv::Point &p2, 
		const float max_affine_shift, const float incr_affine_shift, const int cols, const int rows, const int r)
	{
		//Create a circular mask of affine shifts to trial
		int radius = (int)std::ceil(max_affine_shift / incr_affine_shift);
		int diam = 2*radius+1;
		cv::Mat affine_shifts = cv::Mat(diam, diam, CV_8UC1, cv::Scalar(0));
		
		//Mark the pixels corresponding to affine transform control point shifts that will be trailed
		cv::circle(affine_shifts, cv::Point(radius, radius), radius, cv::Scalar(1), -1, 8, 0);

		//Calculate the affine transform shift set
		std::vector<float> shift_set(diam);
		for (int i = 0; i < diam; i++)
		{
			shift_set[i] = (i - radius) * max_affine_shift / radius;
		}

		//Positions of the affine transform control points on the source images
		cv::Point2f srcTri[3];
		srcTri[0] = cv::Point2f( 0, 0 );
		srcTri[1] = cv::Point2f( c1.cols-1, 0 );
		srcTri[2] = cv::Point2f( radius, c1.rows-1 );

		//Calculate the improvements in matching for various combinations of the affine shifts
		byte *bi, *bj, *bk;
        #pragma omp parallel for
		//Shift about control point i
		for (int i1 = 0; i1 < affine_shifts.rows; i1++)
		{
			bi = affine_shifts.ptr<byte>(i1);
			for (int i2 = 0; i2 < affine_shifts.cols; i2++)
			{
				//If the pixel is marked as a shift to perform
				if (bi[i2])
				{
					//Shift about control point j
					for (int j1 = 0; j1 < affine_shifts.rows; j1++)
					{
						bj = affine_shifts.ptr<byte>(j1);
						for (int j2 = 0; j2 < affine_shifts.cols; j2++)
						{
							//If the pixel is marked as a shift to perform
							if (bj[j2])
							{
								//Shift about control point k
								for (int k1 = 0; k1 < affine_shifts.rows; k1++)
								{
									bk = affine_shifts.ptr<byte>(k1);
									for (int k2 = 0; k2 < affine_shifts.cols; k2++)
									{
										//If the pixel is marked as a shift to perform
										if (bk[k2])
										{
											//Destination control points
											cv::Point2f dstTri[3];
											dstTri[0] = srcTri[0] + cv::Point2f(shift_set[i2], shift_set[i1]);
											dstTri[1] = srcTri[1] + cv::Point2f(shift_set[j2], shift_set[j1]);
											dstTri[2] = srcTri[2] + cv::Point2f(shift_set[k2], shift_set[k1]);

											//Get the Affine Transform
											cv::Mat warp_mat = getAffineTransform( srcTri, dstTri );

											//Apply the Affine Transform just found to the second spot
											cv::Mat w = cv::Mat(c2.size(), c2.type());
											cv::warpAffine( c2, w, warp_mat, w.size(), cv::INTER_LANCZOS4 );

											//Get the new overlapping region for this affine transform
											cv::Mat affine_overlap = get_affine_overlap_mask( warp_mat, mask, co.P1, r, co.P2, r, cols, rows );

											//Prepare vectors to store the values of the matrices in the overlapping region
											int nnz_px = cv::countNonZero(affine_overlap);
											std::vector<float> vals1(nnz_px);
											std::vector<float> vals2(nnz_px);

											//Get the pixel value and distances from circle centers for pixels in the overlapping region
											byte *b;
											for (int i = 0, k = 0; i < affine_overlap.rows; i++)
											{
												b = affine_overlap.ptr<byte>(i);
												for (int j = 0; j < affine_overlap.cols; j++, k++)
												{
													//If the pixel is marked as an overlapping region pixel
													if (b[j])
													{
														//Get the values of the pixels
														vals1[k] = c1.at<float>(i-p1.y, j-p1.x);
														vals2[k] = c2.at<float>(i-p2.y, j-p2.x);
													}
												}
											}

											double pear = pearson_corr(vals1, vals2);

											//Continue later if needed...
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	/*Create a matrix indicating where a spot overlaps with an affinely transformed spot
	**Inputs:
	**warp_mat: cv::Mat &, Affine warp matrix
	**mask: cv::Mat &, Mask indicating the spot overlap in the absence of affine transformation
	**P1: cv::Point2d, Center of one of the circles
	**r1: const int &, Radius of one of the circles
	**P2: cv::Point2d, Center of the other circle
	**r2: const int &, Radius of the other circle
	**cols: const int, Number of columns in the mask
	**rows: const int, Number of rows in the mask
	**val: const byte, value to set the mask elements. Defaults to 255
	**Returns:
	**cv::Mat, 8 bit image where the overlapping region is marked with ones
	*/
	cv::Mat get_affine_overlap_mask(cv::Mat &warp_mat, cv::Mat &mask, cv::Point2d P1, const int r1, cv::Point2d P2,
		const int r2, const int cols, const int rows, const byte val)
	{
		//Create separate masks to draw the circles on
		cv::Mat circle1 = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));
		cv::Mat circle2 = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));

		//Draw first circle
		{
			//Get the minimum and maximum rows and columns to iterate between
			int min_col, max_col, min_row, max_row;
			min_col = std::max(0, (int)(P1.x-r1));
			max_col = std::min(cols-1, (int)(P1.x+r1));
			min_row = std::max(0, (int)(P1.y-r1));
			max_row = std::min(rows-1, (int)(P1.y+r1));

			//Iterate accross the circle rows
			byte *p;
            #pragma omp parallel for
			for (int i = min_row, rel_row = min_row-P1.y; i <= max_row; i++, rel_row++)
			{
				//Create C-style pointers to interate across the circle with
				p = circle1.ptr<byte>(i);

				//Get columns to iterate between
				int c = (int)std::sqrt(r1*r1-rel_row*rel_row);
				int min = std::max(min_col, (int)(P1.x-c));
				int max = std::min(max_col, (int)(P1.x+c));

				//Iterate across columns
				for (int j = min; j <= max; j++)
				{
					p[j] = val;
				}
			}
		}

		//Draw second circle
		{
			//Get the minimum and maximum rows and columns to iterate between
			int min_col, max_col, min_row, max_row;
			min_col = std::max(0, (int)(P2.x-r2));
			max_col = std::min(cols-1, (int)(P2.x+r2));
			min_row = std::max(0, (int)(P2.y-r2));
			max_row = std::min(rows-1, (int)(P2.y+r2));

			//Iterate accross the circle rows
			byte *p;
            #pragma omp parallel for
			for (int i = min_row, rel_row = min_row-P2.y; i <= max_row; i++, rel_row++)
			{
				//Create C-style pointers to interate across the circle with
				p = circle2.ptr<byte>(i);

				//Get columns to iterate between
				int c = (int)std::sqrt(r2*r2-rel_row*rel_row);
				int min = std::max(min_col, (int)(P2.x-c));
				int max = std::min(max_col, (int)(P2.x+c));

				//Iterate across columns
				for (int j = min; j <= max; j++)
				{
					p[j] = val;
				}
			}
		}

		//Affinely transform the second circle
		cv::Mat w = cv::Mat(circle2.size(), circle2.type());
		cv::warpAffine( circle2, w, warp_mat, w.size(), cv::INTER_LANCZOS4 );

		//Erode the edges of the transform to compensate for edge effects
		cv::erode(w, w, cv::Mat(), cv::Point(-1,-1), 1);

		//The overlapping region is where the circle and the affinely transformed circle are marked, within
		//the confines of the original region to ensure that nothing has gone off the edge of the image
		return circle1 & w & mask;
	}

	/*Get the relative positions of overlapping regions of spots using the ORB feature detector
	**Inputs:
	**groups: std::vector<cv::Mat> &, Preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	**group_pos: std::vector<cv::Point> &, Positions of top left corners of circles' bounding squares
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**grouped_idx: std::vector<std::vector<int>> &, Groups of consecutive image indices where the spots are all in the same position
	**is_in_img: std::vector<bool> &, True when the spot is in the image so that indices can be grouped
	**radius: const int, Radius of the spots
	**col_max: const int, Maximum column difference between spot positions
	**row_max: const int, Maximum row difference between spot positions
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	*/
	void overlap_rel_pos(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam)
	{
		//Calculate the length scale of features in the image
		//FLAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG

		//Construct ORB feature matcher to perform ORB operations in the loop
		cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 2, 1, 0, 0, 3, cv::ORB::HARRIS_SCORE, 10);

		//Get the spots that are fully in images
		std::vector<std::vector<double>> rel_overlap_pos;
		std::vector<cv::Vec2i> combinations;
		for (int k = 0, m = 0, counter = 0; k < grouped_idx.size(); k++)
		{
			if (is_in_img[k])
			{
				for (int l = k+1, n = m+1; l < grouped_idx.size(); l++)
				{
					if (is_in_img[l])
					{
						//Find the overlapping region between the circles
						circ_overlap co = get_overlap(spot_pos, rel_pos, col_max, row_max, radius, grouped_idx[m][0], grouped_idx[n][0], 
							cols, rows);

						//Check if they overlap and the overlapping region is fully in the image
						if (co.overlap)
						{
							//Find if the overlapping region is entirely on the image
							if (on_img(co.bounding_rect[0], cols, rows) && on_img(co.bounding_rect[1], cols, rows))
							{
								//Create a mask idicating where the circles overlap to extract the values from the images
								cv::Mat co_mask = gen_circ_overlap_mask(co.P1, radius, co.P2, radius, cols, rows);

								//Get the number of overlapping pixels from the mask
								int num_overlap = cv::countNonZero(co_mask);

								//Check that there are enough pixels to make a meaningful estimate of the symmetry center
								if (num_overlap > MIN_OVERLAP_PX_NUM)
								{
									//Parameters describing each overlap. By index: 0 - Fraction of circle radius from the first circle's center,
									//1 - Fraction of circle radius from the second circle's center, 2 - ratio of the first circle's pixel value
									//to the second circle's
									std::vector<cv::Vec3d> overlaps(num_overlap);

									//Get the ratios of the overlapping protions of the image
									cv::Vec2f shift; //Shift of the second image relative to the first
									get_overlap_rel_pos(co_mask, groups[m], groups[n], group_pos[m], group_pos[n], orb, num_overlap, shift);
								}
							}
						}

						//Go to th next group
						n++;
					}
				}

				//Go to next group
				m++;
			}
		}
	}

	/*Relative position of one spot overlapping with another using the ORB feature detector
	**Inputs:
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate where the circles overlap
	**c1: cv::Mat &, One of the circles
	**c2; cv::Mat &, The other circles
	**p1: cv::Point &, Top left point of a square containing the first circle on the detector
	**p2: cv::Point &, Top left point of a square containing the second circle on the detector
	**max_shift: const float, Maximum shift to consider
	**incr_shift: const float, Steps to adjust the shift by
	**nnz: const int, Number of non-zero pixels in the mask. The number of features looked for will be based on this
	**shift: cv::Vec2f &, Output how much the second image needs to be shifted to align it
	*/
	void get_overlap_rel_pos(cv::Mat &mask, cv::Mat &c1, cv::Mat &c2, cv::Point &p1, cv::Point &p2,
		cv::Ptr<cv::ORB> &orb, const int nnz, cv::Vec2f &shift)
	{
		//Create an image to store ratios on and a mask indicating them
		cv::Mat overlap1 = cv::Mat(c1.size(), CV_32FC1, cv::Scalar(0.0));
		cv::Mat overlap2 = cv::Mat(c1.size(), CV_32FC1, cv::Scalar(0.0));

		//Get the pixel value and distances from circle centers for pixels in the overlapping region
		byte *b;
		for (int i = 0, co_num = 0; i < mask.rows; i++)
		{
			b = mask.ptr<byte>(i);
			for (int j = 0; j < mask.cols; j++)
			{
				//If the pixel is marked as an overlapping region pixel
				if (b[j])
				{
					//Get the values of the pixels
					double val1, val2;
					val1 = c1.at<float>(i-p1.y, j-p1.x);
					val2 = c2.at<float>(i-p2.y, j-p2.x);

					overlap1.at<float>(i-p1.y, j-p1.x) = val1;
					overlap2.at<float>(i-p1.y, j-p1.x) = val2;
				}
			}
		}

		//Make number of keypoints to look for proportional to the area
		int num_feat = 20;//nnz / OVERLAP_REL_POS_PX_PER_KEYPOINT;

		//Use the keypoints to find the relative position of one overlap to the other
		orb->setMaxFeatures(num_feat);

		//Convert images to 8 bit
		cv::Mat norm1, norm2;
		cv::normalize(overlap1, norm1, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::normalize(overlap2, norm2, 0, 255, cv::NORM_MINMAX, CV_8UC1);

		//Detect keypoints in both of the images
		std::vector<cv::KeyPoint> keypoints1, keypoints2;
		cv::Mat descriptors1, descriptors2;
		orb->detectAndCompute(norm1, mask*255, keypoints1, descriptors1);
		orb->detectAndCompute(norm2, mask*255, keypoints2, descriptors2);

		//Match the features
		std::vector<cv::DMatch> matches;
		cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create( cv::NORM_HAMMING2, true );

		matcher->match( descriptors1, descriptors2, matches );

		cv::Mat img;
		cv::drawMatches( norm1, keypoints1, norm2, keypoints2, matches, img );

		display_CV(img);
	}
}