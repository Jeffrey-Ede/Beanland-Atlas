#include <bright_field_sym.h>

namespace ba
{
	/*Get the position of the central spot's symmetry center, whatever it's symmetry may be
	**Inputs:
	**img: cv::Mat &, Rotated central survey so that the symmetry center is easier to extract
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**num_spots: const int, Number of equidistant spots on the aligned diffraction pattern
	**Returns:
	**cv::Point, Position of the symmetry center
	*/
	cv::Point2f center_sym_pos(cv::Mat &img, std::vector<float> &angles, const int num_spots)
	{
		//Rotate the bright field survey to produce a stack of images that can be compared
		std::vector<cv::Mat> surveys(num_spots);
		surveys[0] = img;
        #pragma omp parallel for
		for (int i = 0; i < num_spots; i++)
		{
			surveys[i] = rotate_CV(img, angles[i]);
		}

		//Get the rotational symmetry
		std::vector<std::vector<float>> rot_sym = get_rotational_between_sym(surveys);

		//Get the mirror symmetry
		std::vector<std::vector<float>> mir_sym = get_mirror_between_sym(surveys);

		////Find the absolute positions of the symmetry centers of the rotated images from their relative positions
		//Eigen::MatrixXf pairs(3, 3); //Combinations of rotated images' separations
		//Eigen::VectorXf dist_rows(3); //Row separations
		//Eigen::VectorXf dist_cols(3); //Column separations

		//								//Dictate the rotation between combinations
		//pairs << -1, 1, 0,
		//	-1, 0, 1,
		//	0, -1, 1;

		////Add the row separations to the vector
		//dist_rows << sym[0][2], sym[1][2], sym[2][2];

		////Add the column separations to the vector
		//dist_cols << sym[0][1], sym[1][1], sym[2][1];

		////Solve to get the row positions of the symmetry centre
		//Eigen::VectorXf rows = pairs.colPivHouseholderQr().solve(dist_rows);

		////Solve to get the columns positions of the symmetry centre
		//Eigen::VectorXf cols = pairs.colPivHouseholderQr().solve(dist_cols);

		return cv::Point(1, 2); //FLAG
	}
}