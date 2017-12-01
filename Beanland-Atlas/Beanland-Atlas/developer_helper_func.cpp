#include <beanland_atlas.h>

/*Display C++ API ArrayFire array
**Inputs:
**arr: af::array &, ArrayFire C++ API array to display
**scale: float, Multiply the array elements by this value before displaying
**plt_name: char *, Optional name for plot
**dim0: int, Optional image side size
**dim1: int, Optional image side size
*/
void display_AF(af::array &arr, float scale, char * plt_name, int dim0, int dim1)
{
	af::Window window(dim0, dim1, plt_name);
	do
	{
		window.image(arr.as(f32)*scale);
	} while( !window.close() );
}

/*Display C API ArrayFire array
**Inputs:
**arr: af_array &, ArrayFire C API array to display
**scale: float, Multiply the array elements by this value before displaying
**plt_name: char *, Optional name for plot
**dim0: int, Optional image side size
**dim1: int, Optional image side size
*/
void display_AF(af_array &arr, float scale, char * plt_name, int dim0, int dim1)
{
	af::Window window(dim0, dim1, plt_name);
	do
	{
		window.image(af::array(arr).as(f32)*scale);
	} while( !window.close() );
}


/*Display OpenCV mat
**Inputs:
**mat: cv::Mat &, OpenCV mat to display
**scale: float, Multiply the mat elements by this value before displaying
**plt_name: char *, Optional name for plot
*/
void display_CV(cv::Mat &mat, float scale, char * plt_name)
{
	//Set up the window to be resizable while keeping its aspect ratio
	cv::namedWindow( plt_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO );

	//Show the OpenCV mat
	cv::imshow( plt_name, mat*scale );
	cv::waitKey(0);
}

/*Print the size of the first 2 dimensions of a C++ API ArrayFire array
**Input:
**arr: af::array &, Arrayfire C++ API array
*/
void print_AF_dims(af::array &arr)
{
	printf("dims = [%lld %lld]\n", arr.dims(0), arr.dims(1));
}

/*Print the size of the first 2 dimensions of a C API ArrayFire array
**Input:
**arr: af_array &, Arrayfire C API array
*/
void print_AF_dims(af_array &arr)
{
	printf("dims = [%lld %lld]\n", af::array(arr).dims(0), af::array(arr).dims(1));
}