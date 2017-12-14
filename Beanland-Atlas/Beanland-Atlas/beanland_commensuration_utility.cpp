#include <beanland_atlas.h>

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

		std::cout << rotated;

		//cv::convertPointsFromHomogeneous(rotated, rotated);
		//display_CV(rotated);

		return rotated;
	}

	/*Estimate the radius of curvature of the Ewald sphere
	**Input:
	**spot_pos: std::vector<cv::Point> &, Positions of spots in the aligned images' average px values diffraction pattern
	**Returns:
	**float, Initial estimate of the Ewald sphere radius of curvature
	*/
	float ewald_radius(std::vector<cv::Point> &spot_pos)
	{
		//Just return the largest possible radius for now, appromiating the surface to be completely flat
		return FLT_MAX;
	}

	/*Difference between overlapping regions of circles
	**Input:


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

	/*Get Find the smallest circles that can be used to construct the each region of the Beanland atlas without gaps
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the images
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**max_radius: the maximum radius of the spots that can be used
	**Returns:
	**std::vector<std::vector<int>>, Circles contributing to each pixel
	*/
	std::vector<std::vector<int>> get_small_spot_sizes(std::vector<std::vector<int>> &rel_pos, std::vector<cv::Point> &spot_pos, 
		const int max_radius)
	{
		std::vector<std::vector<int>> contributors(rel_pos[0].size());

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

		return contributors;
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
	**relative to the the spot
	*/
	std::vector<std::vector<std::vector<int>>> get_spot_overlaps(std::vector<std::vector<int>> &rel_pos,
		std::vector<std::vector<int>> &grouped_idx, const float &max_dist)
	{
		//Assign memory to store the relative positions of all the spots
		std::vector<std::vector<std::vector<int>>> group_rel_pos(grouped_idx.size());

		//Compare the positions of each consecutive group of same position spots...
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
			group_rel_pos.push_back(rel_pos_overlappers);
		}

		return group_rel_pos;
	}
}