#include <beanland_atlas.h>

namespace ba
{
	/*Get the surveys made by spots equidistant from the central spot in the aligned images average px values diffraction pattern. These 
	**will be used to identify the atlas symmetry
	**Inputs:
	**positions: std::vector<cv::Point> &, Relative positions of the individual spots in the aligned images average px values diffraction 
	**pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**and still be considered to be one of the equidistant spots
	**Returns:
	**struct equidst_surveys, Indices of the spots nearest to and equidistant from the brightest spot and their angles to a horizontal
	**line drawn horizontally through the brightest spot
	*/
	struct equidst_surveys equidistant_surveys(std::vector<cv::Point> &spot_pos, float threshold)
	{
		//Find the minimum distance from the central spot
		int min_dst = INT_MAX;
        #pragma omp parallel for reduction(min:min_dst)
		for (int i = 1; i < spot_pos.size(); i++)
		{
			//Get position relative to the central spot
			int dx = spot_pos[i].x - spot_pos[0].x;
			int dy = spot_pos[i].y - spot_pos[0].y;

			//Check if the distance of the spot is smaller than the minimum
			if (std::sqrt(dx*dx + dy*dy) < min_dst)
			{
				//Make distance the new minimum
				min_dst = std::sqrt(dx*dx + dy*dy);
			}
		}

		//Find the indices of the surveys equidistant from the central spot, within the threshold
		std::vector<int> equidst_surveys;
		std::vector<float> survey_angles;
		for (int i = 1; i < spot_pos.size(); i++)
		{
			//Get position relative to the central spot
			int dx = spot_pos[i].x - spot_pos[0].x;
			int dy = spot_pos[i].y - spot_pos[0].y;

			//Check if the distance of the spot is smaller than the threshold
			if (std::sqrt(dx*dx + dy*dy) < (1.0+threshold)*min_dst)
			{
				//Note the position's index
				equidst_surveys.push_back(i);

				//Get the angle to the horizontal drawn through the brightest spot
				float angle = std::acos(dx / std::sqrt(dx*dx + dy*dy));
				survey_angles.push_back(dy > 0 ? angle : 2*PI-angle);
			}
		}

		struct equidst_surveys survey_param;
		survey_param.indices = equidst_surveys;
		survey_param.angles = survey_angles;

		return survey_param;
	}

	/*Identifies the symmetry of the Beanland Atlas using Fourier analysis and Pearson normalised product moment correlation
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**threshold: float, Maximum proportion of the distance between the brightest spot and the spot least distant from it that a spot can be
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, float threshold, bool cascade)
	{
		//Get the indices of the surveys to compare to identify the atlas symmetry and the angles of the spots used to create surveys
		//relative to a horizontal line drawn through the brightest spot
		struct equidst_surveys equidst = equidistant_surveys(spot_pos, threshold);

		//Order the indices into order order of increasing position angle from the horizontal
		order_indices_by_angle(equidst.angles, equidst.indices);

		//Rotate the surveys so that they all have the same angle relative to a horizontal line drawn through the brightest spot
		std::vector<cv::Mat> rot_to_align = rotate_to_align(surveys, equidst.angles, equidst.indices);

		//Special case based on the number of spots equidistant from the straight through beam
		if (equidst.indices.size() == 2)
		{
			//2 spot symmetry support will not be added until it is needed as additional symmetries have to be checked for and there is
			//no immediate demand
			return atlas_sym_2(rot_to_align, spot_pos, cascade);
		}
		else
		{
			//3 nearest spots on the aligned diffraction pattern
			if (equidst.indices.size() == 3)
			{
				return atlas_sym_3(rot_to_align, spot_pos, cascade);
			}
			//4 or 6 nearest spots on the aligned diffraction pattern
			else
			{
				return atlas_sym_4_or_6(rot_to_align, spot_pos, cascade);
			}
		}
	}

	/*Calculate the symmetry of a 2 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_2(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		atlas_sym a;

		return a;
	}

	/*Calculate the symmetry of a 3 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_3(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		atlas_sym a;

		return a;
	}

	/*Calculate the symmetry of a 4 or 6 survey atlas
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots on the aligned average image values diffraction pattern
	**cascade: bool, If true, calculate Pearson normalised product moment correlation coefficients for all possible symmetries for a given
	**number of surveys. If false, the calculation will be slightly faster
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym atlas_sym_4_or_6(std::vector<cv::Mat> &rot_to_align, std::vector<cv::Point> &spot_pos, bool cascade)
	{
		//Get the center of symmetry of the central spot
		cv::Point center_sym_pos(rot_to_align[0]);

		//Mirror symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> mirror = get_mirror_between_sym(rot_to_align);

		//Rotation symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> rot_between = get_mirror_between_sym(rot_to_align);

		//Internal rotational symmetry Pearson normalised product moment coefficient spectrums in surveys
		std::vector<std::vector<float>> rot_in = get_rotational_in_sym(rot_to_align);

		/* Get the average Pearson product moment correlation coefficients for each of the symmetries */
		
		//Caclulate the average mirror symmetry quantifications
		float mean_mir_between = 0.0f;
		float mean_mir_in = 0.0f;
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			mean_mir_in += mirror[k][0];
			k++;

			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				mean_mir_between += mirror[k][0];
			}
		}
		mean_mir_in /= rot_to_align.size();
		mean_mir_between /= (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;

		//Mean rotational symmetry between surveys quantification
		float mean_rot_between = 0.0f;
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				mean_rot_between += rot_between[k][0];
			}
		}
		mean_rot_between /= (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;

		//Mean rotational symmetry in surveys quntification
		float mean_rot_in = 0.0f;
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			mean_rot_in += rot_in[i][0];
		}
		mean_rot_in /= rot_to_align.size();

		//Internal rotational mirror symmetry Pearson normalised product moment coefficient spectrums in surveys
		std::vector<std::vector<float>> mir_rot_in = get_mir_rot_in_sym(rot_to_align, 50);

		atlas_sym a;

		return a;
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

		return dst;
	}

	/*Rotate the surveys so that they are all aligned at the same angle to a horizontal line drawn through the brightest spot
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by individual spots, some of which will be compared to identify the symmetry 
	**group
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	**Returns:
	**std::vector<cv::Mat>, Images rotated so that they are all aligned. 
	*/
	std::vector<cv::Mat> rotate_to_align(std::vector<cv::Mat> &surveys, std::vector<float> &angles, std::vector<int> &indices)
	{
		//Rotate each mat so that they all have the same orientation to a horizontal line drawn through the brightest spot in the aligned patter
		std::vector<cv::Mat> rot_to_align(indices.size());
        #pragma omp parallel for
		for (int i = 0; i < indices.size(); i++)
		{
			rot_to_align[i] = rotate_CV(surveys[indices[i]], RAD_TO_DEG*angles[i]);
		}

		return rot_to_align;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys and reflections of the other surveys in
	**a mirror line between the 2 surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and relative row and column shift of the 
	**second image, in that order
	*/
	std::vector<std::vector<float>> get_mirror_between_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Use sum of squares formula to get the number of comparisons
		int num_comp = rot_to_align.size() * (rot_to_align.size()+1) / 2;

		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(num_comp);
		
		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//...the higher index matrices
			for (int j = i; j < rot_to_align.size(); j++, k++)
			{
				//Flip the second mat in the comparison
				cv::Mat mir;
				cv::flip(rot_to_align[j], mir, 1);

				//Quantify the symmetry and record the shift of highest symmetry
				sym_param[k] = quantify_rel_shift(rot_to_align[i], mir);
			}
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients between the surveys when rotated to the positions of the other
	**surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_between_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Use the Hockey-Stick formula to get the number of comparisons
		int num_comp = (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;

		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(num_comp);

		//Compare each matrix with...
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			//...the higher index matrices and itself
			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				//Quantify the symmetry and record the shift of highest symmetry
				sym_param[k] = quantify_rel_shift(rot_to_align[i], rot_to_align[j]);
			}
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for 180 deg rotational symmetry in the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_rotational_in_sym(std::vector<cv::Mat> &rot_to_align)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(rot_to_align.size());

		//Look for rotational symmetry in each matrix
        #pragma omp parallel for
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Rotate the matrix 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i], rot_180, -1);

			//Quantify the symmetry and record the shift of highest symmetry
			sym_param[i] = quantify_rel_shift(rot_to_align[i], rot_180);
		}

		return sym_param;
	}

	/*Calculate Pearson nomalised product moment correlation coefficients for mirror rotational symmetry inside the surveys
	**Inputs:
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<std::vector<float>>, Pearson normalised product moment correlation coefficients and centres of symmetry, in that order
	*/
	std::vector<std::vector<float>> get_mir_rot_in_sym(std::vector<cv::Mat> &rot_to_align, float mir_row)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<std::vector<float>> sym_param(rot_to_align.size());

		//ROI row
		int row = mir_row > (int)mir_row ? (int)std::ceil(mir_row) : (int)mir_row+1;
		row--;

		//Look for rotational symmetry in each matrix
        #pragma omp parallel for
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Partition matrix into 2 parts, along the known mirror line
			cv::Rect roi_left = cv::Rect(0, 0, rot_to_align[i].cols, row);
			cv::Rect roi_right = cv::Rect(0, row, rot_to_align[i].cols, rot_to_align[i].rows - row);

			//Rotate the rightmost region of interest 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i](roi_right), rot_180, -1);

			//Mirror the rotated region of interest
			cv::Mat mir_rot;
			cv::flip(rot_180, mir_rot, 1);

			//Quantify the symmetry and record the shift of highest symmetry
			sym_param[i] = quantify_rel_shift(rot_to_align[i](roi_left), mir_rot);
		}

		return sym_param;
	}

	/*Get the position of the central spot's symmetry center, whatever it's symmetry may be
	**Inputs:
	**img: cv::Mat &, Rotated central survey so that the symmetry center is easier to extract
	**num_equidst: const int, Number of equidistant spots on the aligned diffraction pattern
	**Returns:
	**cv::Point, Position of the symmetry center
	*/
	cv::Point center_sym_pos(cv::Mat &rot_survey, int num_spots)
	{
		//Since we just want the center of symmetry, we only have to check the rotational symmetry
		std::vector<cv::Mat> surveys(num_spots);
		surveys[0] = rot_survey[0];
		for (int i = 1; i < num_spots; i++)
		{
			surveys[i] = rotate_CV(rot_survey, i * 2*PI / num_spots);
		}

		std::vector<std::vector<float>> sym = get_rotational_between_sym(surveys);

		//All possible 3-fold symmetries have rotational symmetry between surveys
		if (num_spots == 3)
		{

		}
		//4 and 6 fold symmetries either have rotational symmetry between every spot (1 apart) or between spots 2 apart
		else
		{

		}

		return cv::Point(sym[0][1], sym[0][2]);
	}
}