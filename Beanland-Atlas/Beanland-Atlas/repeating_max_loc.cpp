#include <repeating_max_loc.h>

namespace ba
{
	/**Calculates the positions of repeating maxima in noisy data. A peak if the data's Fourier power spectrum is used to 
	**find the number of times the pattern repeats, assumung that the data only contains an integer number of repeats.
	**This number of maxima are then searched for.
	**Inputs:
	**corr: std::vector<float>, Noisy data to look for repeatis in
	**num_angles: int, Number of elements in data
	**pos_mir_sym: std::array<int, 4>, Numbers of maxima to look for. Will choose the one with the highest power spectrum value
	**Returns:
	**std::vector<int>, Idices of the maxima in the array. The number of maxima is the size of the array
	*/
	std::vector<int> repeating_max_loc(std::vector<float> corr, int num_angles, std::array<int, 4> pos_mir_sym){

		double *angles;
		fftw_complex *fft_corr_result;
		fftw_plan fft_corr;

		//Allocate memory to store result
		angles = (double*)fftw_malloc(num_angles * sizeof(double));
		for (int k = 0; k < num_angles; k++) {
			angles[k] = (double)corr[k];
		}
		fft_corr_result = (fftw_complex*)fftw_malloc((num_angles/2+1)*sizeof(fftw_complex));

		//Create plan
		fft_corr = fftw_plan_dft_r2c_1d(num_angles, angles, fft_corr_result, FFTW_ESTIMATE);

		//Execute plan
		fftw_execute(fft_corr);

		//Symmetry is given by the power spectrum component with the highest amplitude
		double max_power = 0;
		int symmetry;
		for(int k = 0; k < pos_mir_sym.size(); k++){

			//Calculate frequency power
			double power = fft_corr_result[pos_mir_sym[k]][0]*fft_corr_result[pos_mir_sym[k]][0]+
				fft_corr_result[pos_mir_sym[k]][1]*fft_corr_result[pos_mir_sym[k]][1];

			if(power > max_power){
				max_power = power;
				symmetry = pos_mir_sym[k];
			}
		}

		//Free fftw resources
		fftw_destroy_plan(fft_corr);
		fftw_free(angles);
		fftw_free(fft_corr_result);

		//Cut the Pearson normalised product moment correlation coefficient spectrum into the same number of chunks as the symmetry
		int size = num_angles/symmetry;
		int rem = num_angles%symmetry;
		std::vector<int> max_pos(symmetry, -1);

		//Increase window size depending of the size of the remainder
		int rem_add = 1;
		if (rem) {
			rem_add = 1;
		}
		else {
			rem_add = 0;
		}

		//Get the position of the first maxima
		int origin = std::distance(corr.begin(), std::max_element(corr.begin(), corr.begin()+size+rem_add));

		//Center cutting window on this maximum to prevent artifacts from maxima being at edge of cutting window
		int size_left = size/2;
		int size_right = size/2+size%2;

		//If there is a window that cuts across the zeroeth index to the end of the array, use cyclicity to handle that first as a special case
		int begin_cuts, end_cuts, idx_offset;
		if (origin > size_left) {

			begin_cuts = 0;
			end_cuts = 1;

			//Find position of maximum in partial windows on both sides of 0 index
			int dist1 = std::distance(corr.begin(), 
				std::max_element(corr.begin()+num_angles+origin-size_left-size, corr.begin()+num_angles));
			int dist2 = std::distance(corr.begin(), std::max_element(corr.begin(), corr.begin()+origin-size_left+rem_add));

			//Order maxima locations from low angle to high angle in case it makes them easier to use later
			if (corr[dist2] >= corr[dist1]) {
				max_pos[0] = dist2;
				idx_offset = 1;
			}
			else {
				max_pos[symmetry-1] = dist1;
				idx_offset = 0;
			}
		}
		else {
			if (origin < size_left) {

				begin_cuts = 1;
				end_cuts = 0;

				//Find position of maximum in partial windows on both sides of 0 index
				int dist1 = std::distance(corr.begin(), 
					std::max_element(corr.begin()+num_angles+origin-size_left, corr.begin()+num_angles));
				int dist2 = std::distance(corr.begin(), std::max_element(corr.begin(), corr.begin()+origin+size_right+rem_add));

				//Order maxima locations from low angle to high angle in case it makes them easier to use later
				if (corr[dist2] >= corr[dist1]) {
					max_pos[0] = dist2;
					idx_offset = 0;
				}
				else {
					max_pos[symmetry-1] = dist1;
					idx_offset = -1;
				}
			}
			else {

				//Window does not cross the 0 index so cyclicity is not invoked
				begin_cuts = 0;
				end_cuts = 0;
				idx_offset = 0;
			}
		}

		int rem_add_prev;
		if (begin_cuts && rem) {
			rem_add_prev = 1;
		}
		else {
			rem_add_prev = 0;
		}

		//Get other Pearson coefficient spectrum maxima
		for (int k = begin_cuts; k < symmetry-end_cuts; k++) {

			//Adjust window size by remainder
			if (k < rem) {
				rem_add = k+1;
			}
			else {
				rem_add = 0;
			}

			//Get position of maxima
			max_pos[k+idx_offset] = std::distance(corr.begin(), 
				std::max_element(corr.begin()+origin-size_left+k*size+rem_add_prev, corr.begin()+origin+size_right+k*size+rem_add));

			rem_add_prev = rem_add;
		}

		return max_pos;
	}
}