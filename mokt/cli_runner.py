import numpy as np
import sys
from termcolor import cprint

from .test_environment import TestEnvironment


def test_kernel_from_file(
        src_path, kernel_name, inputs, expected_outputs, global_size,
        local_size):
    """Executes an OpenCL kernel from file, verifies its output, and prints stats.

    This is a convenience wrapper around a `TestEnvironment`. It is recommended
    that you read documentation for methods in `TestEnvironment`
    before using this method.

    Args:
        src_path (str): Path to the C source file.
        kernel_name (str): Name of the kernel function,
            as defined in the C source.
        inputs (list): See documentation for `TestEnvironment.run_kernel`.
        expected_outputs (list) See documentation for
            `TestEnvironment.run_kernel` and `TestEnvironment.output_defs`.
        global_size (tuple): See documentation for `TestEnvironment.run_kernel`.
        local_size (tuple | None): See documentation for `TestEnvironment.run_kernel`.
    """

    env = TestEnvironment()
    kernel = env.kernel_from_file(src_path, kernel_name)

    exec_event, actual_outputs = env.run_kernel(
        kernel, inputs, env.output_defs(expected_outputs), global_size,
        local_size)

    verify_and_profile(exec_event, inputs, expected_outputs, actual_outputs)


def verify_and_profile(exec_event, inputs, expected_outputs, actual_outputs):
    exec_time = 1e-3 * (exec_event.profile.end - exec_event.profile.start)
    print(f'Accelerator execution time: {exec_time:.1f}us')

    # TODO: Print deltas
    for i, (expected, actual) in enumerate(zip(expected_outputs,
                                               actual_outputs)):
        if np.allclose(expected, actual):
            cprint(
                f'[Output #{i}]: Expected and actual values are equal',
                'green')
        else:
            cprint(
                f'[Output #{i}]: Actual values do not match expected', 'red')
            deltas = []
            for j in range(len(expected)):
                if expected[j] != actual[j]:
                    error = max(expected[j], actual[j]) - min(expected[j], actual[j])
                    deltas.append(error)
                    if (j < 20):
                        try:
                            print('%d) in: %f, %f = tf: %f, test: %f, delta = %f' % (j, inputs[0][j], inputs[1][j], expected[j], actual[j], error))
                        except IndexError:
                            print('%d) in: %f = tf: %f, test: %f, delta = %f' % (j, inputs[0][j], expected[j], actual[j], error))
            print('average delta: %f' % (np.sum(deltas) / len(deltas)))
        
