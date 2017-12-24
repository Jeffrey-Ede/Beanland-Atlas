#pragma once

#include <defines.h>
#include <includes.h>

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
	std::vector<int> repeating_max_loc(std::vector<float> corr, int num_angles, std::array<int, 4> pos_mir_sym);
}