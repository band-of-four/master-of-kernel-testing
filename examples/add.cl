__kernel void add(
	__global const float *first_input, __global const float *second_input, __global float *output) {

	int global_id = get_global_id(0);

	output[global_id] = first_input[global_id] + second_input[global_id];

}