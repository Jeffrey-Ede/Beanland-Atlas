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
		float ewald_rad;

		return ewald_rad;
	}
}