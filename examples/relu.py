import sys
sys.path.append('.')

from mokt import TestEnvironment, TestData

env = TestEnvironment()


@TestData(
    tf_checkpoint_dir='../../Documents/resnet50v1_traindir',
    tf_values={
        'input': 'v/tower_0/cg/resnet_v10/add:0',
        'output': 'v/tower_0/cg/resnet_v10/Relu:0'
    })
def test_relu(test_data):
    relu_in = test_data['input'].flatten()[:20]
    expected_out = test_data['output'].flatten()[:20]

    kernel = env.create_program(
        """
        __kernel void relu(
            __global const float *a_g, __global float *res_g)
        {
          int gid = get_global_id(0);
          res_g[gid] = max(a_g[gid], 0.0f);
        }
        """).relu

    exec_event, outputs = env.run_kernel(
        kernel, [relu_in], [(expected_out.shape[0], expected_out.dtype.type)],
        global_size=relu_in.shape,
        local_size=(20, ))
    actual_out = outputs[0]

    print(relu_in)
    print(expected_out)
    print(actual_out)


test_relu()
