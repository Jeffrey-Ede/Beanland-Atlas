inline void AtomicAdd(volatile global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel
void freq_spectrum1D(
    __global float* input,
    __global float* histogram,
    int width,
    int half_width,
    float inv_width2,
    float inv_height2,
    float inv_max_freq,
    size_t num_bins)
{
    int i = get_global_id(0);

    //Get distances from center of shifted 2D fft
    int y = i%(half_width+1);
    int x = i/(half_width+1);
    if (x > half_width)
    {
        x = width - x;
    }
    x += 1;

    //Add result to histogram, restricting index to appropriate range
    int idx = (int)(sqrt(x*x*inv_width2 + y*y*inv_height2)*inv_max_freq*num_bins);
    if (idx < num_bins){
        AtomicAdd(&histogram[idx], input[i]);
    }
    else
    {
        AtomicAdd(&histogram[num_bins-1], input[i]);
    }
}