#define inv_root_2pi 0.3989422804

__kernel
void gauss_kernel_extended(
    __global float* outputImage,
    float inv_sigma,
    float minus_half_inv_sigma2,
    int rows_half_width,
    int cols_half_width,
    int full_width)
{
    int i = get_global_id(0);

    int x = (i%full_width)-rows_half_width;
    int y = (i/full_width)-cols_half_width;

    outputImage[i] = inv_sigma*inv_root_2pi * exp(minus_half_inv_sigma2*(x*x+y*y));
}