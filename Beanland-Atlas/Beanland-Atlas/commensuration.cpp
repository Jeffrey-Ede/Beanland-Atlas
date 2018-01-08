#include <commensuration.h>

namespace ba
{
	/*Calculate the condenser lens profile using the overlapping regions of spots
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual floating point images that have been stereographically corrected to extract spots from
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**col_max: int, Maximum column difference between spot positions
	**row_max: int, Maximum row difference between spot positions
	**radius: const int, Radius about the spot locations to extract pixels from
	**Returns:
	**std::vector<cv::Mat>, Dynamical diffraction effect decoupled Bragg profile
	*/
	std::vector<cv::Mat> condenser_profile(std::vector<cv::Mat> &mats, cv::Point2d &spot_pos, std::vector<std::vector<int>> &rel_pos,
		int col_max, int row_max, int radius)
	{
		std::cout << "In condenser_profile()" << std::endl;

		//Find the non-consecutively same position spots and record the indices of multiple spots with the same positions
		std::vector<std::vector<int>> grouped_idx = consec_same_pos_spots(rel_pos);

		//Create collection of the spot images at different positions, averaging the consecutively same position spots.
		int diam = 2*radius+1;
		std::vector<cv::Mat> groups;
		std::vector<cv::Point> group_pos;
		std::vector<bool> is_in_img;
		grouping_preproc(mats, grouped_idx, spot_pos, rel_pos, col_max, row_max, radius, diam, groups, group_pos, is_in_img);

		//Get the condenser lens profile
		cv::Mat profile = get_condenser_lens_profile(groups, group_pos, spot_pos, rel_pos, grouped_idx, is_in_img, radius, 
			col_max, row_max, mats[0].cols, mats[0].rows);

		std::vector<cv::Mat> something;
		return something;
	}

	/*Identify groups of consecutive spots that all have the same position
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots in the image stack
	**Returns:
	**std::vector<std::vector<int>>, Groups of spots with the same location
	*/
	std::vector<std::vector<int>> consec_same_pos_spots(std::vector<std::vector<int>> &rel_pos)
	{
		//Find consecutive spots with the same position
		std::vector<std::vector<int>> groups;
		int first_same = 0; //Start of same position train
		for (int i = 1; i < rel_pos[0].size(); i++)
		{
			//Check if relative position is different from the previous
			if (rel_pos[0][i] != rel_pos[0][i-1] || rel_pos[1][i] != rel_pos[1][i-1])
			{
				//Create the train of same position indices
				std::vector<int> same_train(i - first_same);
				for (int j = 0; j < same_train.size(); j++)
				{
					same_train[j] = first_same + j;
				}

				//Append the indices sequence of the same train
				groups.push_back(same_train);

				//Restart the same train
				first_same = i;
			}
		}

		//Create the last train of same position indices
		std::vector<int> same_train(rel_pos[0].size() - first_same);
		for (int j = 0; j < same_train.size(); j++)
		{
			same_train[j] = first_same + j;
		}

		//Append the indices sequence of the same train
		groups.push_back(same_train);

		//Return the groups of same position indices
		return groups;
	}

	/*Extract a Bragg peak from an image stack, averaging the spots that are in the same position in consecutive images
	**Inputs:
	**mats: std::vector<cv::Mat> &, Images to extract the spots from
	**grouped_idx: std::vector<std::vector<int>> &, Groups of consecutive image indices where the spots are all in the same position
	**spot_pos: cv::Point2d &, Position of the spot on the aligned images average px values diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots in the input images to the first image
	**col_max: const int &, Maximum column difference between spot positions
	**row_max: const int &, Maximum row difference between spot positions
	**radius: const int &, Radius of the spot
	**diam: const int &, Diameter of the spot
	**groups: std::vector<cv::Mat> &, Output preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	**group_pos: std::vector<cv::Point> &, Positions of top left corners of circles' bounding squares
	**is_in_img: std::vector<bool> &, Output to mark true when the spot is in the image so that indices can be grouped
	*/
	void grouping_preproc(std::vector<cv::Mat> &mats, std::vector<std::vector<int>> &grouped_idx, cv::Point2d &spot_pos, 
		std::vector<std::vector<int>> &rel_pos, const int &col_max, const int &row_max, const int &radius,
		const int &diam, std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, std::vector<bool> &is_in_img)
	{
		groups.clear();
		group_pos.clear();
		is_in_img = std::vector<bool>(grouped_idx.size());

		//For each group of spots...
		for (int i = 0; i < grouped_idx.size(); i++)
		{
			//Index of the first image in the group
			int j = grouped_idx[i][0];

			is_in_img[i] = spot_pos.y >= row_max-rel_pos[1][j] && spot_pos.y < row_max-rel_pos[1][j]+mats[j].rows &&
				spot_pos.x >= col_max-rel_pos[0][j] && spot_pos.x < col_max-rel_pos[0][j]+mats[j].cols;

			//Check if the spot is in the image. If the first in the group isn't, none of them are as they are all in the same position
			if (is_in_img[i])
			{
				//Accumulate the mats in the group for averaging
				cv::Mat acc = cv::Mat(diam, diam, CV_32FC1, cv::Scalar(0.0));

				//...extract the group of spots, averaging them together if there are multiple in the group
				for (int k = 0; k < grouped_idx[i].size(); k++)
				{
					//Index of image in the image stack
					j = grouped_idx[i][k];

					//Accumulate the circle in the accumulator
					accumulate_circle(mats[j], spot_pos.x-col_max+rel_pos[0][j], spot_pos.y-row_max+rel_pos[1][j], radius,
						acc, radius, radius);
				}

				//Store the preprocessed Bragg peak
				groups.push_back( acc / grouped_idx[i].size() );
				group_pos.push_back( cv::Point( spot_pos.x-col_max+rel_pos[0][j]-radius, spot_pos.y-row_max+rel_pos[1][j]-radius ) );
			}
		}
	}

	/*Extracts a circle of data from an OpenCV mat and accumulates it in another mat.
	**Inputs:
	**mat: cv::Mat &, Reference to a floating point OpenCV mat to extract the data from
	**col: const int &, column of circle origin
	**row: const int &, row of circle origin
	**rad: const int &, radius of the circle to extract data from
	**acc: cv::Mat &, Floating point OpenCV mat to accumulate the data in
	**acc_col: const int &, Column of the accumulator mat to position the circle at
	**acc_row: const int &, Row of the accumulator mat to position the circle at
	*/
	void accumulate_circle(cv::Mat &mat, const int &col, const int &row, const int &rad, cv::Mat &acc, const int &acc_col,
		const int &acc_row)
	{
		//Get the minimum and maximum rows and columns to iterate between
		int min_col, max_col, min_row, max_row;
		min_col = std::max(0, col-rad);
		max_col = std::min(mat.cols-1, col+rad);
		min_row = std::max(0, row-rad);
		max_row = std::min(mat.rows-1, row+rad);

		//Iterate accross the circle rows
		float *p, *q;
        #pragma omp parallel for
		for (int i = min_row, rel_row = min_row-row, k = acc_row-rad; i <= max_row; i++, rel_row++, k++)
		{
			//Create C-style pointers to interate across the circle with
			p = mat.ptr<float>(i);
			q = acc.ptr<float>(k);

			//Get columns to iterate between
			int c = (int)std::sqrt(rad*rad-rel_row*rel_row);
			int min = std::max(min_col, col-c);
			int max = std::min(max_col, col+c);

			//Iterate across columns
			for (int j = min, l = acc_col-c; j <= max; j++, l++)
			{
				q[l] += p[j];
			}
		}
	}

	/*Calculate an initial estimate for the dark field decoupled Bragg profile using the preprocessed Bragg peaks. This function is redundant.
	**It remains in case I need to generate data from it for my thesis, etc. in the future
	**Input:
	**blur_not_consec: std::vector<cv::Mat>> &, Preprocessed Bragg peaks. Consecutive Bragg peaks in the same position have been averaged
	**and they have been Gaussian blurred to remove unwanted high frequency noise
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**Returns:
	**cv::Mat, Dark field decouple Bragg profile of the accumulation
	*/
	cv::Mat get_acc_bragg_profile(std::vector<cv::Mat> &blur_not_consec, cv::Mat &circ_mask)
	{
		//Assign memory to store the dark field decoupled Bragg profile estimate
		cv::Mat profile = cv::Mat(blur_not_consec[0].size(), CV_32FC1, cv::Scalar(0.0));

		//Get the maximum pixel in every column
		byte *c;

		//Iterate across mask rows...
		for (int i = 0; i < circ_mask.rows; i++)
		{
			//...and columns
			c = circ_mask.ptr<byte>(i);
			for (int j = 0; j < circ_mask.cols; j++)
			{
				//Get maximum values for mask pixels
				if (c[j])
				{
					//Iterate through every preprocessed Bragg peak to find the maximum
					float max = 0;
					for (int k = 0; k < blur_not_consec.size(); k++)
					{
						//Check if the pixel value is higher than the maximum
						if (blur_not_consec[k].at<float>(i, j) > max)
						{
							max = blur_not_consec[k].at<float>(i, j);
						}
					}

					//Record the maximum on the Bragg profile estimate
					profile.at<float>(i, j) = max;
				}
			}
		}

		return profile;
	}

	/*Use the unique overlaps of each group of spots to determine the condenser lens profile
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
	cv::Mat get_condenser_lens_profile(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows)
	{
		//Information from overlapping circle pixels needed to determine the condenser lens profile
		std::vector<cv::Vec3d> overlap_px_info;

		//Get the spots that are fully in images
		for (int k = 0, m = 0; k < grouped_idx.size(); k++)
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

								//Parameters describing each overlap. By index: 0 - Fraction of circle radius from the first circle's center,
								//1 - Fraction of circle radius from the second circle's center, 2 - ratio of the first circle's pixel value
								//to the second circle's
								std::vector<cv::Vec3d> overlaps(num_overlap);

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

											std::cout << i-group_pos[m].y << ", " << j-group_pos[m].x << ", " << val1 << std::endl;
											std::cout << i-group_pos[n].y << ", " << j-group_pos[n].x << ", " << val2 << std::endl;

											overlaps[co_num++] = cv::Vec3d(dist1, dist2, val1/val2);
										}
									}
								}

								std::cout << "********************" << std::endl;

								//Append the overlapping pixel information to the collation
								overlap_px_info.insert(overlap_px_info.end(), overlaps.begin(), overlaps.end());
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

		std::cout << "here..." << std::endl;
		for (int i = 0; i < overlap_px_info.size(); i++)
		{
			std::cout << i << ": " << overlap_px_info[i][0] << ", "  << overlap_px_info[i][1] << " | " << overlap_px_info[i][2] << std::endl;
		}
		std::getchar();

		//Fit cubic Bezier curve to the radial profile using least squares minimisation
		cv::Mat cubic_Bezier;

		return cubic_Bezier;
	}

	/*Calculates the center of and the 2 points closest to and furthest away from the center of the overlapping regions 
	**of 2 spots
	**Inputs:
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images to first image
	**col_max: const int &, Maximum column difference between spot positions
	**row_max: const int &, Maximum row difference between spot positions
	**radius: const int, Radius of the spots
	**m: const int, Index of one of the images to compare
	**n: const int, Index of the other image to compare
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	**Returns:
	**circ_overlap, Structure describing the region where the circles overlap
	*/
	circ_overlap get_overlap(cv::Point2d &spot_pos, std::vector<std::vector<int>> &rel_pos,	const int &col_max,
		const int &row_max, const int radius, const int m, const int n, const int cols, const int rows)
	{
		//Get the positions of the spots in each image
		circ_overlap co;
		co.P1 = cv::Point2d( spot_pos.x-col_max+rel_pos[0][m], spot_pos.y-row_max+rel_pos[1][m] );
		co.P2 = cv::Point2d( spot_pos.x-col_max+rel_pos[0][n], spot_pos.y-row_max+rel_pos[1][n] );

		//Check that there is overlap between these 2 circles
		double dist = std::sqrt((co.P1.x-co.P2.x)*(co.P1.x-co.P2.x) + (co.P1.y-co.P2.y)*(co.P1.y-co.P2.y));
		co.overlap = dist < 2*radius;
		if (co.overlap)
		{
			//Calculate the center
			co.center = 0.5 * (co.P1 + co.P2);

			//Calculate the positions at minimal distances from the overlap center
			std::vector<cv::Point2d> minima(2);
			minima[0] = co.P1 + (1.0 - radius/dist) * (co.P2 - co.P1);
			minima[1] = co.P2 - (1.0 - radius/dist) * (co.P2 - co.P1);
			co.minima = minima;

			//Calculate the positions at maximal distances from the overlap center
			std::vector<cv::Point2d> maxima(2);
			double delta = 0.5 * std::sqrt( (dist + 2*radius) * dist * dist * (2*radius - dist) );
			maxima[0] = co.center + cv::Point2d( delta * (co.P1.y - co.P2.y) / (dist*dist), 
				-delta * (co.P1.x - co.P2.x) / (dist*dist));
			maxima[1] = co.center + cv::Point2d( -delta * (co.P1.y - co.P2.y) / (dist*dist), 
				delta * (co.P1.x - co.P2.x) / (dist*dist));
			co.maxima = maxima;

			//Find the points describing the rectangle bounding the extrema
			double xl, yl, xu, yu;
			if (co.P1.x > co.P2.x)
			{
				xl = co.P1.x - radius;
				xu = co.P2.x + radius;
			}
			else
			{
				xl = co.P2.x - radius;
				xu = co.P1.x + radius;
			}
			if (co.P1.y > co.P2.y)
			{
				yl = co.P1.y - radius;
				yu = co.P2.y + radius;
			}
			else
			{
				yl = co.P2.y - radius;
				yu = co.P1.y + radius;
			}

			//Record 2 points to describe the bounding rectangle
			std::vector<cv::Point> bounds(2);
			bounds[0] = cv::Point((int)std::floor(xl), (int)std::floor(yl));
			bounds[1] = cv::Point((int)std::ceil(xu), (int)std::ceil(yu));
			co.bounding_rect = bounds;
		}

		return co;
	}

	/*Boolean indicating whether or not a point lies on an image
	**Inputs
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	**point: cv::Point, Point to be checked
	**Returns:
	**bool, If true, the point is on the image
	*/
	bool on_img(cv::Point point, const int cols, const int rows)
	{
		return point.x >= 0 && point.y >= 0 && point.x < cols && point.y < rows;
	}

	/*Boolean indicating whether or not a point lies on an image
	**Inputs
	**img: cv::Mat &, Image to check if the point is on
	**point: cv::Point, Point to be checked
	**Returns:
	**bool, If true, the point is on the image
	*/
	bool on_img(cv::Mat &img, cv::Point point)
	{
		return point.x >= 0 && point.y >= 0 && point.x < img.cols && point.y < img.rows;
	}

	/*Boolean indicating whether or not a point lies on an image
	**Inputs
	**img: cv::Mat &, Image to check if the point is on
	**col: const int, Column of point to be checked
	**row: const int, Row of point to be checked
	**Returns:
	**bool, If true, the point is on the image
	*/
	bool on_img(cv::Mat &img, const int col, const int row)
	{
		return col >= 0 && row >= 0 && col < img.cols && row < img.rows;
	}

	/*Generate a mask where the overlapping region between 2 circles is marked by ones
	**Inputs:
	**P1: cv::Point2d, Center of one of the circles
	**r1: const int &, Radius of one of the circles
	**P2: cv::Point2d, Center of the other circle
	**r2: const int &, Radius of the other circle
	**cols: const int, Number of columns in the mask
	**rows: const int, Number of rows in the mask
	**val: const byte, value to set the mask elements. Defaults to 1
	**Returns:
	**cv::Mat, 8 bit image where the overlapping region is marked with ones
	*/
	cv::Mat gen_circ_overlap_mask(cv::Point2d P1, const int r1, cv::Point2d P2, const int r2, const int cols,
		const int rows, const byte val)
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
		
		//The overlapping region is where both circles are marked
		return circle1 & circle2;
	}

	/*Overload the << operator to print circ_overlap structures
	**Inputs:
	**os: std::ostream &, Reference to the operating system,
	**co: const circ_overlap &, Overlapping circle region structure
	**Returns:
	**std::ostream &, Reference to the operating system
	*/
	std::ostream & operator<<(std::ostream & os, const circ_overlap & co)
	{
		os << "Overlap: " << co.overlap << std::endl;
		os << "Overlap center: " << co.center << std::endl;
		os << "Minima: " << co.minima[0] << ", " << co.minima[1] << std::endl;
		os << "Maxima: " << co.maxima[0] << ", " << co.maxima[1] << std::endl;
		os << "Centres P1, P2: " << co.P1 << ", " << co.P2 << std::endl;
		os << "Bounding rectangle: " << co.bounding_rect[0] << ", " << co.bounding_rect[1] << std::endl;

		return os;
	}
}