__kernel void relu(
    __global const float *input, __global float *output) {

  int global_id = get_global_id(0);

  output[global_id] = max(input[global_id], 0.0f);

}
