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
	**frac_for_sym: const float, Mean Pearson product moment correlation coefficients for each symmetry must be at least this fraction of 
	**the maximum Pearson coefficient for that symmetry to be considered present
	**Returns:
	**struct atlas_sym, Atlas symmetries
	*/
	struct atlas_sym identify_symmetry(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, const float threshold,
		const float frac_for_sym)
	{
		//Get the indices of the surveys to compare to identify the atlas symmetry and the angles of the spots used to create surveys
		//relative to a horizontal line drawn through the brightest spot
		struct equidst_surveys equidst = equidistant_surveys(spot_pos, threshold);

		//Order the indices into order order of increasing position angle from the horizontal
		order_indices_by_angle(equidst.angles, equidst.indices);

		//Rotate the surveys so that they all have the same angle relative to a horizontal line drawn through the brightest spot
		std::vector<cv::Mat> rot_to_align = rotate_to_align(surveys, equidst.angles, equidst.indices);

		//Mirror symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> mirror = get_mirror_sym(rot_to_align);

		//Rotation symmetry Pearson normalised product moment coefficient spectrums between surveys
		std::vector<std::vector<float>> rot_between = get_rotational_between_sym(rot_to_align);

		//Internal rotational symmetry Pearson normalised product moment coefficient spectrums in surveys
		std::vector<std::vector<float>> rot_in = get_rotational_in_sym(rot_to_align);

		//Caclulate the average mirror symmetry quantifications
		float mean_mir_between = 0.0f;
		float mean_mir_in = 0.0f;
		float mean_mir_between_2 = 0.0f; //Only present for 4 or 6 equidistant spots
		for (int i = 0, k = 0; i < rot_to_align.size(); i++)
		{
			mean_mir_in += mirror[k][0];
			k++;

			for (int j = i+1; j < rot_to_align.size(); j++, k++)
			{
				mean_mir_between += mirror[k][0];
			}

			for (int j = i+2; j < rot_to_align.size(); j += 2, k += 2)
			{
				mean_mir_between_2 += mirror[k][0];
			}
		}
		mean_mir_in /= rot_to_align.size();
		mean_mir_between /= (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 3;
		mean_mir_between_2 /= (rot_to_align.size()+1) * rot_to_align.size() * (rot_to_align.size()-1) / 6;

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

		//Decide which symmetries are present
		float max_pear = std::max(mean_mir_in, std::max(mean_mir_between, std::max(mean_mir_between_2, 
			std::max(mean_rot_between, mean_rot_in))));

		//Record the symmetries present. By index: 0 - mir_in, 1 - mir_between, 2 - mir_between_2, 3 - rot_between, 4 - rot_in, 
		//5 - mirror rotational symmetry (will be calculated later)
		std::vector<bool> symmetries(6);
		symmetries[0] = mean_mir_in > frac_for_sym * max_pear;
		symmetries[1] = mean_mir_between > frac_for_sym * max_pear;
		symmetries[2] = mean_mir_between_2 > frac_for_sym * max_pear;
		symmetries[3] = mean_rot_between > frac_for_sym * max_pear;
		symmetries[4] = mean_rot_in > frac_for_sym * max_pear;

		//If there is mirror symmetry between and in every survey, get the internal rotational mirror symmetry Pearson normalised 
		//product moment coefficient spectrums in surveys
		if (symmetries[1] && symmetries[1])
		{
			std::vector<std::vector<float>> mir_rot_in = get_mir_rot_in_sym(rot_to_align, refl_lines);
			symmetries[5] = mean_mir_rot_in > frac_for_sym * max_pear;
		}
		//There is only mirror rotational symmetry in the surveys when there is mirror symmetry between and in every survey
		else
		{
			symmetries[5] = false;
		}

		//Solve an overconstrained system of simultaneous equations to get the symmetry centres
		std::vector<cv::Point2f> sym_centers = get_sym_centers(mirror, rot_between, rot_in, symmetries, rot_to_align.size(), rot_to_align);

		//Get the center of symmetry of the central spot
		//cv::Point2f sym_center = center_sym_pos(surveys[0], rot_to_align.size()); //Probably not useful


		//Special case based on the number of spots equidistant from the straight through beam
		//if (equidst.indices.size() == 2)
		//{
		//	//2 spot symmetry support will not be added until it is needed as additional symmetries have to be checked for and there is
		//	//no immediate demand
		//	return atlas_sym_2(rot_to_align, spot_pos, cascade);
		//}
		//else
		//{
		//	//3 nearest spots on the aligned diffraction pattern
		//	if (equidst.indices.size() == 3)
		//	{
		//		return atlas_sym_3(rot_to_align, spot_pos, cascade);
		//	}
		//	//4 or 6 nearest spots on the aligned diffraction pattern
		//	else
		//	{
		//		return atlas_sym_4_or_6(rot_to_align, spot_pos, cascade);
		//	}
		//}
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


		atlas_sym a;

		return a;
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
	std::vector<std::vector<float>> get_mirror_sym(std::vector<cv::Mat> &rot_to_align)
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
				cv::flip(rot_to_align[j], mir, 0);

				display_CV(rot_to_align[j]);
				display_CV(mir);

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
	**refl_lines: std::vector<int> &, Lines to perform reflections in
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficient for internal mirror rotation symmetry
	*/
	std::vector<float> get_mir_rot_in_sym(std::vector<cv::Mat> &rot_to_align, std::vector<int> &refl_lines)
	{
		//Assign vectors to hold the symmetry parameters
		std::vector<float> sym_param(rot_to_align.size());

		//Look for rotational symmetry in each matrix
        #pragma omp parallel for
		for (int i = 0; i < rot_to_align.size(); i++)
		{
			//Partition matrix into 2 parts, along the known mirror line
			cv::Rect roi_left = cv::Rect(0, 0, rot_to_align[i].cols, refl_lines[i]);
			cv::Rect roi_right = cv::Rect(0, refl_lines[i], rot_to_align[i].cols, rot_to_align[i].rows - refl_lines[i]);

			//Rotate the rightmost region of interest 180 degrees
			cv::Mat rot_180;
			cv::flip(rot_to_align[i](roi_right), rot_180, -1);

			cv::Mat left = rot_to_align[i](roi_left);

			//Quantify the symmetry
			std::vector<float> sym(3);
			sym_param[i] = pearson_corr(rot_180, left,  rot_to_align[i].rows - 2*refl_lines[i], 0);
		}

		return sym_param;
	}

	/*Get the centers of symmetry in each of the surveys from the quantified symmetries by solving the overdetermined set of simultaneous
	**equations created during symmetry registration
	**Inputs:
	**mirror: std::vector<std::vector<float>> &, Mirror symmetry quantification
	**rot_between: std::vector<std::vector<float>> &, Rotational symmetry between surveys quantification
	**rot_in: std::vector<std::vector<float>> &, 180 deg rotational symmetry in surveys quantification
	**symmetries: std::vector<bool> &, Symmetries present. By index: 0 - mir_in, 1 - mir_between, 2 - mir_between_2, 3 - rot_between, 
	**4 - rot_in. Other symmetries, such as rotational mirror symmetry, are not used by this function
	**num_surveys: const int, Number of surveys
	**rot_to_align: std::vector<cv::Mat> &, Surveys that have been rotated so that they are all at the same angle to a horizontal line
	**drawn through the brightest spot
	**Returns:
	**std::vector<cv::Point2f>, Centers of symmetry for each of the surveys (column, then row)
	*/
	std::vector<cv::Point2f> get_sym_centers(std::vector<std::vector<float>> &mirror, std::vector<std::vector<float>> &rot_between,
		std::vector<std::vector<float>> &rot_in, std::vector<bool> &symmetries, const int num_surveys, std::vector<cv::Mat> &rot_to_align)
	{
		//Number of inter-comparisons between surveys for between symmetries
		int num_inter_comp = (num_surveys+1) * num_surveys * (num_surveys-1) / 3;

		//Get the number of equations
		int num_eqn;
		num_eqn = symmetries[0] ? num_surveys : 0;
		num_eqn += symmetries[1] ? 2*num_inter_comp : 0;
		num_eqn += symmetries[1] ? 0 : (symmetries[2] ? num_inter_comp : 0); //Only use mirrors between spots 2 apart if there is none 1 apart
		num_eqn += symmetries[3] ? num_inter_comp : 0;
		num_eqn += symmetries[4] ? num_surveys : 0;

		//Create simultaneous equations describing the symmetry centers
		Eigen::MatrixXf eqns = Eigen::MatrixXf::Constant(num_eqn, 2*num_surveys, 0.0f); //Express the coordinates of surveys being compared...
		Eigen::VectorXf diff(num_eqn); //...in terms of the differences in symmetry center positions of the surveys or otherwise

		//Add equations describing mirror symmetry in surveys
		if (symmetries[0])
		{
			for (int i = 0, k = 0; i < num_surveys; i++)
			{
				//Row estimate
				eqns(i, i) = 1.0f;
				diff(i) = (rot_to_align[0].rows - mirror[k][2]) / 2;
				k += i;
			}
		}

		//Keep track of how many equations have beed added to the matrix
		int eqn_num = num_surveys;

		//Add equations describing mirror symmetry between surveys
		if (symmetries[1])
		{
			for (int i = 0, k = 0; i < num_surveys; i++)
			{
				k++;

				for (int j = i+1; j < num_surveys; j++, k++, eqn_num++)
				{
					//Row estimate
					eqns(eqn_num, j) = 1.0f;
					eqns(eqn_num, i) = -1.0f;
					diff(eqn_num) = mirror[k][2];
					
					eqn_num++;

					//Column estimate
					eqns(eqn_num, j+num_surveys) = 1.0f;
					eqns(eqn_num, i+num_surveys) = -1.0f;
					diff(eqn_num) = mirror[k][1];
				}
			}
		}
		//Mirror symmetry between every second survey, not between all surveys
		else
		{
			if (symmetries[2])
			{
				for (int i = 0, k = 0; i < num_surveys; i++)
				{
					k += 2;

					for (int j = i+2; j < num_surveys; j += 2, k += 2, eqn_num++)
					{
						//Row estimate
						eqns(eqn_num, j) = 1.0f;
						eqns(eqn_num, i) = -1.0f;
						diff(eqn_num) = mirror[k][2];

						eqn_num++;

						//Column estimate
						eqns(eqn_num, j+num_surveys) = 1.0f;
						eqns(eqn_num, i+num_surveys) = -1.0f;
						diff(eqn_num) = mirror[k][1];
					}
				}
			}
		}

		//Rotational symmetry between surveys
		if (symmetries[3])
		{
			for (int i = 0, k = 0; i < num_surveys; i++)
			{
				for (int j = i+1; j < num_surveys; j++, k++, eqn_num++)
				{
					//Row estimate
					eqns(eqn_num, j) = 1.0f;
					eqns(eqn_num, i) = -1.0f;
					diff(eqn_num) = rot_between[k][2];

					eqn_num++;

					//Column estimate
					eqns(eqn_num, j+num_surveys) = 1.0f;
					eqns(eqn_num, i+num_surveys) = -1.0f;
					diff(eqn_num) = rot_between[k][1];
				}
			}
		}

		if (symmetries[4])
		{
			for (int i = 0; i < num_surveys; i++, eqn_num++)
			{
				//Row estimate
				eqns(eqn_num, i) = 1.0f;
				diff(eqn_num) = (rot_to_align[0].rows - rot_in[i][2]) / 2;

				eqn_num++;

				//Column estimate
				eqns(eqn_num, i+num_surveys) = 1.0f;
				diff(eqn_num) = (rot_to_align[0].cols - rot_in[i][1]) / 2;
			}
		}

		//Solve the simultaneous equations to find the symmetry center
		Eigen::VectorXf solution = eqns.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(diff);

		//Express the symmetry centers in an easier to use form
		std::vector<cv::Point2f> sym_centers(num_surveys);
		for (int i = 0; i < num_surveys; i++)
		{
			//Column, then row
			sym_centers[i] = cv::Point2f(solution(i+num_surveys), solution(i));
		}

		return sym_centers;
	}

	///*Get the position of the central spot's symmetry center, whatever it's symmetry may be
	//**Inputs:
	//**img: cv::Mat &, Rotated central survey so that the symmetry center is easier to extract
	//**num_equidst: const int, Number of equidistant spots on the aligned diffraction pattern
	//**Returns:
	//**cv::Point, Position of the symmetry center
	//*/
	//cv::Point2f center_sym_pos(cv::Mat &img, int num_spots)
	//{
	//	//Since we just want the center of symmetry, we only have to check the rotational symmetry
	//	std::vector<cv::Mat> surveys(num_spots);
	//	surveys[0] = img;
	//	for (int i = 1; i < num_spots; i++)
	//	{
	//		surveys[i] = rotate_CV(img, i * 2*PI / num_spots);
	//	}

	//	std::vector<std::vector<float>> sym = get_rotational_between_sym(surveys);

	//	//All possible 3-fold symmetries have rotational symmetry between surveys
	//	if (num_spots == 3)
	//	{
	//		//Find the absolute positions of the symmetry centers of the rotated images from their relative positions
	//		Eigen::MatrixXf pairs(3, 3); //Combinations of rotated images' separations
	//		Eigen::VectorXf dist_rows(3); //Row separations
	//		Eigen::VectorXf dist_cols(3); //Column separations

	//									  //Dictate the rotation between combinations
	//		pairs << -1, 1, 0,
	//			-1, 0, 1,
	//			0, -1, 1;

	//		//Add the row separations to the vector
	//		dist_rows << sym[0][2], sym[1][2], sym[2][2];

	//		//Add the column separations to the vector
	//		dist_cols << sym[0][1], sym[1][1], sym[2][1];

	//		//Solve to get the row positions of the symmetry centre
	//		Eigen::VectorXf rows = pairs.colPivHouseholderQr().solve(dist_rows);

	//		//Solve to get the columns positions of the symmetry centre
	//		Eigen::VectorXf cols = pairs.colPivHouseholderQr().solve(dist_cols);
	//	}
	//	//4 and 6 fold symmetries always have rotational symmetry between spots 2 apart
	//	else
	//	{

	//	}

	//	return cv::Point(sym[0][1], sym[0][2]);
	//}
}