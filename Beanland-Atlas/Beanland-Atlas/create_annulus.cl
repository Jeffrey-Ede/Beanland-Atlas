__kernel
void create_annulus(
    __global float* outputImage,
    int inner_rad_squared,
    int outer_rad_squared,
    int half_width,
    int half_height,
    int full_width)
{
    int i = get_global_id(0);

    int x = (i%full_width)-half_width;
    int y = (i/full_width)-half_height;

    //Set pixels between the inner and outer radii to 1.0; set other pixels to 0.0
    if(x*x+y*y >= inner_rad_squared && x*x+y*y <= outer_rad_squared){
        outputImage[i] = 1.0f;
    }
    else{
        outputImage[i] = 0.0f;
    }
}