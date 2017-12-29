#include <correct_distortions.h>

namespace ba
{
	/*Use the known atlas symmetry to detect symmetries in the outer k-space surveys
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**symmetries: std::vector<bool> &, Symmetries present. By index: 0 - mir_in, 1 - mir_between, 2 - mir_between_2, 3 - rot_between, 
	**4 - rot_in, 5 - mir_in_rad
	**min_area_frac: const float, Minimum fraction of the bright field survey's area that a survey must have to be considered
	*/
	void get_further_out_survey_sym(std::vector<cv::Mat> &surveys, std::vector<bool> &symmetries, const float min_area_frac)
	{


		//Estimate positions of symmetry centers in outer spots if there is internal 2mm mirror symmetry or internal rotational symmetry
		if ((symmetries[0] && symmetries[5]) || symmetries[4])
		{

		}
	}

	/*Use Beanland atlas symmetries to correct atlas distortions by applying 
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**varying_rad_between: const float, Vary the positions of the points being perspective transformed between surveys within this radius of their
	**expected positions in the absence of distortion
	**varying_rad_in: const float, Vary the positions of the points being perspective transformed in surveys within this radius of their
	**expected positions in the absence of distortion
	**varying_frac: const float, Vary positions of the points being perspective transformed using steps given by this fraction of the
	**varying radius
	*/
	void correct_distortions(std::vector<cv::Mat> &surveys, const float varying_rad_between, const float varying_rad_in,
		const float varying_frac)
	{
		float varying_step_between = varying_frac * varying_rad_between;
		float varying_step_in = varying_frac * varying_rad_in;

		//Compare supposedly symmetrical regions to determine warps
        #pragma omp parallel for
		for (int i = 0; i < surveys.size(); i++)
		{
			//Use the average feature size to determine the size of the regions to compare
			float feature_size = get_avg_feature_size(surveys[i]);

			//Find the position of highest Pearson product moment correlation
			//cv::Mat pear;
			//cv::matchTemplate(pad1, src2, pear, CV_TM_CCOEFF);
		}
	}
}