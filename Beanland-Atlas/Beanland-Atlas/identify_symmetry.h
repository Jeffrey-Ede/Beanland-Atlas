#pragma once

#include <defines.h>
#include <includes.h>

#include <ident_sym_utility.h>
#include <utility.h>

namespace ba
{
	//Minimum fraction of the maximum mean Pearson normalised product moment correlation coefficient that a mean Pearson coefficient can be for
	//it's symmetry to be registered as present. A more sophisticated metric will be devised later
    #define FRAC_FOR_SYM 0.5

	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
	//for internal rotational symmetry
    #define INTERAL_ROT_SSD_FRAC 0.7

	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
	//for internal mirror symmetry where the flip is perpendicular to radially outwards direction
    #define INTERAL_MIR0_SSD_FRAC 0.7

	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
	//for internal mirror symmetry where the flip is in the radially outwards direction
    #define INTERAL_MIR1_SSD_FRAC 0.7

	/*Get the surveys made by spots equidistant from the central spot in the aligned images average px values diffraction pattern. These 
	**will be used to identify the atlas symmetry
	**Inputs:
	**positions: std::vector<cv::Point> &, Relative positions of the individual spots in the aligned images average px values diffraction 
	**pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**and still be considered to be one of the equidistant spots
	**indices: std::vector<int> &, Output indices of the spots nearest to and equidistant from the brightest spot 
	**angles: std::vector<int> &, Output angles of the spots to a horizontal line drawn horizontally through the brightest spot
	*/
	void equidistant_surveys(std::vector<cv::Point> &spot_pos, float threshold, std::vector<int> &indices, std::vector<float> &angles);

	//Custom data structure to hold atlas symmetry
	struct atlas_sym_struct {
		//parameters
		int x;
	};
	typedef atlas_sym_struct atlas_sym;

	/*Identifies the symmetry of the Beanland Atlas using Fourier analysis and Pearson normalised product moment correlation
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**frac_for_sym: const float, Mean Pearson product moment correlation coefficients for each symmetry must be at least this fraction of 
	**the maximum Pearson coefficient for that symmetry to be considered present
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	atlas_sym identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, const float threshold,
		const float frac_for_sym);

	/*Rotate the surveys so that they are all aligned at the same angle to a horizontal line drawn through the brightest spot
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**std::vector<cv::Mat>, Images rotated so that they are all aligned. 
	*/
	std::vector<cv::Mat> rotate_to_align(std::vector<cv::Mat> &surveys, std::vector<float> &angles, std::vector<int> &indices);

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of the other surveys in
	**a mirror line between the 2 surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_between_sym(std::vector<cv::Mat> &rot_to_align);

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of themselves perpendicular
	**to the radially outwards direction
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_in_sym(std::vector<cv::Mat> &rot_to_align);

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of themselves parallel
	**to the radially outwards direction
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_in_rad_sym(std::vector<cv::Mat> &rot_to_align);

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys when rotated to the positions of the other
	**surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_between_sym(std::vector<cv::Mat> &rot_to_align);

	/*Calculate Pearson nomalised product moment correlation coefficients for 180 deg rotational symmetry in the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**est_sym_centers: std::vector<cv::Point2f> &, Estimated symmetry center positions
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_in_sym(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point2f> &est_sym_centers);

	/*Calculate Pearson nomalised product moment correlation coefficients for mirror rotational symmetry inside an image. This function
	**is not currently being used as this symmetry does not need to be detected in surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**refl_lines: std::vector<int> &, Lines to perform reflections in
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficient for internal mirror rotation symmetry
	*/
	std::vector<float> get_mir_rot_in_sym(std::vector<cv::Mat> &rot_to_align, std::vector<int> &refl_lines);

	/*Get the centers of symmetry in each of the surveys from the quantified symmetries by solving the overdetermined set of simultaneous
	**equations created during symmetry registration
	**Inputs:
	**mirror: std::vector<std::vector<float>> &, Mirror symmetry quantification
	**rot_between: std::vector<std::vector<float>> &, Rotational symmetry between surveys quantification
	**rot_in: std::vector<std::vector<float>> &, 180 deg rotational symmetry in surveys quantification
	**symmetries: std::vector<bool> &, Symmetries present. By index: 0 - mir_in, 1 - mir_between, 2 - mir_between_2, 3 - rot_between, 
	**4 - rot_in, 5 - rot_in_rad
	**num_surveys: const int, Number of surveys
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<cv::Point2f>, Centers of symmetry for each of the surveys (column, then row)
	*/
	std::vector<cv::Point2f> get_sym_centers(std::vector<std::vector<float>> &mirror, std::vector<std::vector<float>> &rot_between,
		std::vector<std::vector<float>> &rot_in, std::vector<bool> &symmetries, const int num_surveys, std::vector<cv::Mat> &rot_to_align);
}