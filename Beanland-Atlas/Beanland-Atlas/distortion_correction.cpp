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
	**Returns:
	**cv::Mat, Condenser lens profile
	*/
	cv::Mat get_distortion_field(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam)
	{
		//Information from overlapping circle pixels needed to determine the condenser lens profile
		std::vector<cv::Vec3d> overlap_px_info;

		//Get the spots that are fully in images
		long int px_tot = 0;
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

								//Check that the number of overlapping pixels is greater than some minimum
								if (num_overlap > MIN_OVERLAP_PX_NUM)
								{
									//Parameters describing each overlap. By index: 0 - Fraction of circle radius from the first circle's center,
									//1 - Fraction of circle radius from the second circle's center, 2 - ratio of the first circle's pixel value
									//to the second circle's
									std::vector<cv::Vec3d> overlaps(num_overlap);

									//Find the affine transform that best commensurates the overlapping spots									
									get_best_overlap(co, co_mask, groups[m], groups[n], group_pos[m], group_pos[n]);

									//Get the pixel value and distances from circle centers for pixels in the overlapping region
									byte *b;
									for (int i = 0, co_num = 0; i < co_mask.rows; i++)
									{
										b = co_mask.ptr<byte>(i);
										for (int j = 0; j < co_mask.cols; j++)
										{
											//If the pixel is marked as an overlapping region pixel
											if (b[j])
											{
												//Get distances from the circle centres
												double dist1, dist2;
												dist1 = std::sqrt((j-co.P1.x)*(j-co.P1.x) + (i-co.P1.y)*(i-co.P1.y));
												dist2 = std::sqrt((j-co.P2.x)*(j-co.P2.x) + (i-co.P2.y)*(i-co.P2.y));

												//Get the values of the pixels
												double val1, val2;
												val1 = groups[m].at<float>(i-group_pos[m].y, j-group_pos[m].x);
												val2 = groups[n].at<float>(i-group_pos[n].y, j-group_pos[n].x);

												overlaps[co_num++] = cv::Vec3d(dist1, dist2, val1/val2);
											}
										}
									}

									//Append the overlapping pixel information to the collation
									overlap_px_info.insert(overlap_px_info.end(), overlaps.begin(), overlaps.end());
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

		//Restructure the data to 3 vectors that can be passed to MATLAB
		std::vector<double> dist1(overlap_px_info.size());
		std::vector<double> dist2(overlap_px_info.size());
		std::vector<double> ratio(overlap_px_info.size());
		for (int i = 0; i < overlap_px_info.size(); i++)
		{
			dist1[i] = overlap_px_info[i][0];
			dist2[i] = overlap_px_info[i][1];
			ratio[i] = overlap_px_info[i][2];
		}

		//Initialise the MATLAB engine
		matlab::data::ArrayFactory factory;
		std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::connectMATLAB();

		//Package data for the cubic Bezier profile calculator
		std::vector<matlab::data::Array> args({
			factory.createArray( { overlap_px_info.size(), 1 }, dist1.begin(), dist1.end() ), //1st distances set
			factory.createArray( { overlap_px_info.size(), 1 }, dist2.begin(), dist2.end() ), //2nd distances set
			factory.createArray( { overlap_px_info.size(), 1 }, ratio.begin(), ratio.end() ), //Intensity ratios
			factory.createScalar<int32_t>(radius),
			factory.createScalar<double>(LS_TOL),
			factory.createScalar<int32_t>(LS_MAX_ITER)
		});

		//Pass data to MATLAB to calculate the cubic Bezier profile
		matlab::data::TypedArray<double> const profile = matlabPtr->feval(
			matlab::engine::convertUTF8StringToUTF16String("bragg_cubic_Bezier"), args);

		//Convert profile to OpenCV mat It is completely symmetrtic so transpositional filling doesn't matter
		cv::Mat bezier_profile = cv::Mat(diam, diam, CV_32FC1);
		{
			int k = 0;
			for (auto val : profile)
			{
				bezier_profile.at<float>( k/diam, k%diam ) = (float)val;
				k++;
			}
		}

		return overlap_px_info;
	}

	/*Calculate the affine transform that best matches the overlapping region between 2 overlapping circles
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

											//Append the overlapping pixel information to the collation
											dist_info.insert(overlap_px_info.end(), overlaps.begin(), overlaps.end());
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
}