#include <beanland_commensuration_utility.h>

namespace ba
{
	/*Rotate an image in into the image plane
	**Input:
	**img: cv::Mat &, Image to rotate
	**angle_horiz: const float &, Angle to rotate into plane from horizontal axis
	**angle_vert: const float &, Angle to rotate into plane from vertical axis
	**Returns:
	**cv::Mat, Image after rotation into the image plane
	*/
	cv::Mat in_plane_rotate(cv::Mat &img, const float &angle_horiz, const float &angle_vert)
	{
		//Create Rodrigues angles vector
		cv::Mat rot_angles = (cv::Mat_<float>(1,3) << 0, angle_horiz, angle_vert);

		//Use the Rodrigues vector to construct the rotation matrix
		cv::Mat rot_mat;
		cv::Rodrigues(rot_angles, rot_mat);

		//Rotate the matrix
		cv::Mat rotated;
		cv::warpPerspective(img, rotated, rot_mat, img.size(), cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);

		//cv::convertPointsFromHomogeneous(rotated, rotated);
		//display_CV(rotated);

		return rotated;
	}

	/*Estimate the radius of curvature of the sample-to-detector sphere
	**Input:
	**spot_pos: std::vector<cv::Point> &, Positions of spots in the aligned images' average px values diffraction pattern
	**xcorr: cv::Mat &, Cross correlation spectrum that's peaks reveal the positions of the spots
	**discard_outer: const int, Spots withing this distance of the boundary are discarded
	**col: const int, column of spot centre
	**row: const int, row of spot centre
	**centroid_size: const int, Size of centroid to take about estimated spot positions to refine them
	**Returns:
	**cv::Vec2f, Initial estimate of the sample-to-detector sphere radius of curvature and average direction, respectively
	*/
	cv::Vec2f get_sample_to_detector_sphere(std::vector<cv::Point> &spot_pos, cv::Mat &xcorr, const int discard_outer, const int col,
		const int row, const int centroid_size)
	{
		//Discard outer spots, if necessary
		std::vector<cv::Point> spots;
		if (discard_outer)
		{
			spots = discard_outer_spots(spot_pos, col, row, discard_outer);
		}
		else
		{
			spots = spot_pos;
		}

		//Get sub-pixely refined positions of the spots, assuming that they are all at least half the centroid away from 
		//centroid weighting spectrum's borders
		std::vector<cv::Point2f> refined_pos = refine_spot_pos(spots, xcorr, col, row, centroid_size);

		/*
		**
		**FINISH THE FUNCTION LATER
		**
		*/

		//Just return the largest possible radius for now, approximating the surface to be completely flat
		return FLT_MAX; //Change to cv::Vec2f later
	}

	/*Refine the estimated positions of spots to sub-pixel accuracy by calculating centroids around their estimated
	**positions based on an image that weights the likelihood of particular pixels representing spots. It is assumed that the spots
	**are at least have the centroid weighting's width and height away from the weighting spectrum's borders
	**Input:
	**spot_pos: std::vector<cv::Point> &, Positions of spots in the aligned images' average px values diffraction pattern
	**xcorr: cv::Mat &, Cross correlation spectrum that's peaks reveal the positions of the spots
	**col: const int, column of spot centre
	**row: const int, row of spot centre
	**centroid_size: const int, Size of centroid to take about estimated spot positions to refine them
	**Returns:
	**std::vector<cv::Point2f>, Sub-pixely accurate spot positions
	*/
	std::vector<cv::Point2f> refine_spot_pos(std::vector<cv::Point> &spot_pos, cv::Mat &xcorr, const int col, const int row,
		const int centroid_size)
	{
		//Assign memory to store the refined spot positions
		std::vector<cv::Point2f> refined_pos(spot_pos.size());

		//Refine the position of each spot by calculating the centroid of the area surrounding it
		int half_size = centroid_size / 2;
		float inv_centroid_area = 1.0f / (float)(centroid_size*centroid_size);
		for (int k = 0; k < spot_pos.size(); k++)
		{
			//Redine the position by calculating the centroid
			float x = 0, y = 0;

			//Iterate over the rows of the centroid weighting square grid...
			for (int i = -half_size; i <= half_size; i++)
			{
				//...and iterate over the columns of the centroid weighting square grid
				for (int j = -half_size; j <= half_size; j++)
				{
					//Add this pixel's contribution to the centroid
					x += xcorr.at<float>(spot_pos[k].y+i, spot_pos[k].x+j) * j;
					y += xcorr.at<float>(spot_pos[k].y+i, spot_pos[k].x+j) * i;
				}
			}

			//Record the value of the centroid
			refined_pos[k] = cv::Point2f(x * inv_centroid_area, y * inv_centroid_area);
		}

		return refined_pos;
	}


	/*Estimate the angular separation between spots using the differences between their overlapping regions
	**Input:
	**img1: cv::Mat &, An image containing a circle or ellipse
	**img2: cv::Mat &, A second image containing a circle or ellipse
	**Returns:
	**cv::Vec2f, Estimated angular position of the second ellipse relative to the first
	*/
	cv::Vec2f est_rel_homography(std::vector<cv::Point> &spot_pos)
	{



		cv::Vec2f angles;
		return angles;
	}

	/*Creat a Beanland atlas survey by using pixels from the nearest spot to each survey position
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the images
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**Returns:
	**cv::Mat, Survey made from the pixels of the nearest spots to each position
	*/
	cv::Mat create_small_spot_survey(std::vector<std::vector<int>> &rel_pos, std::vector<cv::Point> &spot_pos, 
		const int &max_radius)
	{
		//I think the easiest way to do this is simply say each spot centre's distance from each point it covers, marking it and the
		//spot number in 2 separate 2D mats. Then, every time another spot is added, it can be checked if it's distance is smaller than
		//the one found already. If so, it becomes the new nearest spot.

		//Note that spots will be elliptical; not circular, after the homographic warp

		//Outside this function, once it is made, need to use it to decouple Bragg peaks from the dark field. This will require some
		//correction after division by the atlas.

		//Note: this function should figure out which spot is closest to each point on the Beanland atlas so that it can be used to
		//calculate that point. The resulting atlas can then be used to decouple the Bragg profiles from the dark field

		//for (int j = 0; j < rel_pos[0].size(); j++)
		//{
		//	//Check if the spot is in the image
		//	if (spot_pos[k].y >= row_max-rel_pos[1][j] && spot_pos[k].y < row_max-rel_pos[1][j]+mats[j].rows &&
		//		spot_pos[k].x >= col_max-rel_pos[0][j] && spot_pos[k].x < col_max-rel_pos[0][j]+mats[j].cols)
		//	{
		//		///Mask to extract spot from micrograph
		//		cv::Mat circ_mask = cv::Mat::zeros(mats[j].size(), CV_8UC1);

		//		//Draw circle at the position of the spot on the mask
		//		cv::Point circleCenter(spot_pos[k].x-col_max+rel_pos[0][j], spot_pos[k].y-row_max+rel_pos[1][j]);
		//		cv::circle(circ_mask, circleCenter, radius+1, cv::Scalar(1), -1, 8, 0);

		//		//Copy the part of the micrograph containing the spot
		//		cv::Mat imagePart = cv::Mat::zeros(mats[j].size(), mats[j].type());
		//		mats[j].copyTo(imagePart, circ_mask);

		//		//Compend spot to map
		//		float *r, *t;
		//		ushort *s;
		//		byte *u;
		//		for (int m = 0; m < indv_num_mappers[k].rows; m++) 
		//		{
		//			r = indv_maps[k].ptr<float>(m);
		//			s = indv_num_mappers[k].ptr<ushort>(m);
		//			t = imagePart.ptr<float>(m);
		//			u = circ_mask.ptr<byte>(m);
		//			for (int n = 0; n < indv_num_mappers[k].cols; n++) 
		//			{
		//				//Add contributing pixels to maps
		//				r[n] += t[n];
		//				s[n] += u[n];
		//			}
		//		}
		//	}
		//}

		cv::Mat something;
		return something;
	}

	/*Commensurate the images using homographic perspective warps, homomorphic warps and intensity rescaling
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots
	**grouped_idx: std::vector<std::vector<int>> &, Spots where they are grouped if they are consecutively in the same position
	**max_dist: const float &, The maximum distance between 2 instances of a spot for the overlap between them to be considered
	**Returns:
	**std::vector<std::vector<std::vector<int>>>, For each group of consecutive spots, for each of the spots it overlaps with, 
	**a vector containing: index 0 - the consecutive group being overlapped, index 1 - the relative column position of the consecutive
	**group that is overlapping relative to the the spot,  index 2 - the relative row position of the consecutive group that is overlapping 
	**relative to the the spot. The nesting is in that order.
	*/
	std::vector<std::vector<std::vector<int>>> get_spot_overlaps(std::vector<std::vector<int>> &rel_pos,
		std::vector<std::vector<int>> &grouped_idx, const float &max_dist)
	{
		//Assign memory to store the relative positions of all the spots
		std::vector<std::vector<std::vector<int>>> group_rel_pos(grouped_idx.size());

		//Compare the positions of each consecutive group of same position spots...
        #pragma omp parallel for
		for (int i = 0; i < grouped_idx.size(); i++)
		{
			//Assign memory to store relative positions of other consecutive groups with significant overlap with this spot
			std::vector<std::vector<int>> rel_pos_overlappers;

			//...to the positions of the other groups
			for (int j = i+1; j < grouped_idx.size(); j++)
			{
				//Distances between the consecutive groups
				int dx = rel_pos[0][grouped_idx[j][0]] - rel_pos[0][grouped_idx[i][0]];
				int dy = rel_pos[1][grouped_idx[j][0]] - rel_pos[1][grouped_idx[i][0]];
				
				//If the distance is smaller than or equal to the maximum allowed separation...
				if (std::sqrt(dx*dx + dy*dy) <= max_dist)
				{
					//...record the relative position of these groups...
					std::vector<int> relative_position(3);
					relative_position[0] = j;
					relative_position[1] = dx;
					relative_position[2] = dy;

					//..and add it to this group of relative positions
					rel_pos_overlappers.push_back(relative_position);
				}
			}

			//Record this group of relative positions
			group_rel_pos[i] = rel_pos_overlappers;
		}

		return group_rel_pos;
	}

	/*Find the differences between two overlapping spots where they overlap. Also output a mask indicating which pixels correspond to overlap
	**in case some of the differences are zero
	**Input:
	**img1: cv::Mat &, Image containing a spot
	**img2: cv::Mat &, A second image containing a spot
	**origin1: cv::Point &, Column and row of the spot centre in the first image, respectively
	**origin2: cv::Point &, Column and row of the spot centre in the second image, respectively
	**dx: const int &, Relative column of the second spot to the first
	**dy: const int &, Relative row of the second spot to the first
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**diff: cv::Mat &, Difference between the 2 matrices when the first is subtracted from the second
	**mask: cv::Mat &, Mask indicating which pixels are differences of the overlapping regions
	*/
	void get_diff_overlap(cv::Mat &img1, cv::Mat &img2, cv::Point &origin1, cv::Point &origin2, const int &dx, const int &dy,
		cv::Mat &circ_mask, cv::Mat &diff, cv::Mat &mask)
	{
		//Initialise the difference and mask mats, making the mask binary to save memory
		diff = cv::Mat(img1.size(), img1.type(), cv::Scalar(0.0)); //Same type as original
		mask = cv::Mat(img1.size(), CV_8UC1, cv::Scalar(0)); //Binary

		//Create pointers for faster interation: p - img1, q - img2, d - diff, m - mask, c - circ_mask, c2 - second pointer to circle mask
		float *p, *q, *d; 
		byte *m, *c, *c2;

		//Create the mask, utilising the already calculated circle mask to speed up calculations
		//Calculate limits
		int lcol = dx >= 0 ? dx : 0;
		int lrow = dy >= 0 ? dy : 0;
		int ucol = img1.cols-dx < img1.cols ? img1.cols-dx: img1.cols; //Upper column number, not index
		int urow = img1.rows-dy < img1.rows ? img1.rows-dx: img1.rows; //Upper row number, not index

		//Interate over mask rows...
		for (int i = lrow, k = 0; i < urow; i++, k++)
		{
			//...and iterate over mask columns
			m = mask.ptr<byte>(i);
			c = circ_mask.ptr<byte>(i);
			c2 = circ_mask.ptr<byte>(k);
			for (int j = lcol, l = 0; j < ucol; j++, l++)
			{
				//If the point is on the first image's spot and the second's, record this as a region to calculate the difference
				m[j] = c[j] && c2[l] ? 1 : 0;
			}
		}

		//Find the differences between the overlapping regions of the spots where they overlap in the first image
		//Interate over mask rows...
		for (int i = 0; i < mask.rows; i++)
		{
			//...and iterate over mask columns
			m = mask.ptr<byte>(i);
			p = img1.ptr<float>(i);
			q = img2.ptr<float>(i);
			d = diff.ptr<float>(i);
			for (int j = 0; j < mask.cols; j++)
			{
				//Calculate differences where the circles overlap
				d[j] = m[j] ? q[j] - p[j] : 0;
			}
		}
	}

	/*Commensurate the images using homographic perspective warps, homomorphic warps and intensity rescaling
	**Input:
	**input: cv::Mat &, Input image to perspective warp
	**inputQuad: cv::point2f [4], 4 input quadilateral or image plane coordinates, from top-left in clockwise order
	**outputQuad: cv::point2f [4], 4 output quadilateral or world plane coordinates, from top-left in clockwise order
	*/
	cv::Mat indv_perspective_warp(cv::Mat &input, cv::Point2f inputQuad[4], cv::Point2f outputQuad[4])
	{
		//Lambda matrix i.e. the perspective warping matrix
		cv::Mat lambda( 2, 4, CV_32FC1 );

		//Output Image
		cv::Mat output;

		// Set the lambda matrix the same type and size as input
		lambda = cv::Mat::zeros( input.rows, input.cols, input.type() );

		// Get the Perspective Transform Matrix i.e. lambda 
		lambda = cv::getPerspectiveTransform( inputQuad, outputQuad );

		// Apply the Perspective Transform just found to the src image
		cv::warpPerspective( input, output, lambda, output.size() );

		//Rescale intensities so that the sum of the intensities in the original and warped image are the same
		double input_tot = cv::sum( input )[0];
		double output_tot = cv::sum( output )[0];

		return ( input_tot / output_tot ) * output;
	}

	/*
	**Input:
	**
	*/
	cv::Mat ellipse_perspective_warp(cv::Mat &input)
	{
		// Input Quadilateral or Image plane coordinates
		cv::Point2f inputQuad[4]; 
		// Output Quadilateral or World plane coordinates
		cv::Point2f outputQuad[4];

		//The 4 points that select quadilateral on the input, from top-left in clockwise order
		//These four pts are the sides of the rect box used as input 
		inputQuad[0] = cv::Point2f( -30,-60 );
		inputQuad[1] = cv::Point2f( input.cols+50,-50 );
		inputQuad[2] = cv::Point2f( input.cols+100,input.rows+50 );
		inputQuad[3] = cv::Point2f( -50,input.rows+50  );

		//The 4 points where the mapping is to be done , from top-left in clockwise order
		outputQuad[0] = cv::Point2f( 0,0 );
		outputQuad[1] = cv::Point2f( input.cols-1,0 );
		outputQuad[2] = cv::Point2f( input.cols-1,input.rows-1 );
		outputQuad[3] = cv::Point2f( 0,input.rows-1 );

		return indv_perspective_warp( input, inputQuad, outputQuad );
	}
}