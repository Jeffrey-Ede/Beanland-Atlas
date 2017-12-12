#pragma once

/*Print contents of vector, then wait for user input to continue. Defaults to printing the entire vector if no print size is specified
**Inputs:
**vect: std::vector<T> &, Vector to print
**num: const int &, Number of elements to print
*/
template <class T> void print_vect(std::vector<T> &vect, const int num)
{
	//Get the number of elements to print. Defaults to printing the whole vector if no number is provided
	int vect_size = vect.size();
	int max = num ? std::min(num, vect_size) : vect_size;

	//Print the specified number of elements
	for (int i = 0; i < max; i++)
	{
		std::cout << vect[i] << std::endl;
	}

	//Wait for user input to continue
	std::getchar();
}