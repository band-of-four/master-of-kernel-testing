import sys
sys.path.append('.')

from mokt import TestData, test_kernel_from_file
from mokt.cli_runner import get_keys

keys = get_keys()

two_inputs = True if keys.get('--second_input') is not None else False

if keys.get('--cl_source') is None:
    kernel_name = keys.get('--output').split('/')[-1][:-2].lower()
    cl_source = kernel_name + '.cl'
else:
    kernel_name = keys['--kernel_name']
    cl_source = keys['--cl_source']

''' 
keys:
--input/--first_input, --second_input*, --output, --cl_source*, --kernel_name*, --file*
'*' - optionally
if you do not specify 'cl_source' and 'kernel name' then they will be identified automatically by the output operation name
if your output ends with '.../Relu:0', then 'cl_source' will be 'relu.cl' and 'kernel_name' - 'relu'
'''


@TestData(
    tf_checkpoint_dir=keys.get('--chkp_dir'),
    tf_values= {
        'input': keys.get('--first_input'),
        'second_input': keys.get('--second_input'),
        'output': keys.get('--output')
    } if two_inputs else {
        'input': keys.get('--input'),
        'output': keys.get('--output')
    })

def test_add(test_data):
    add_first_in = test_data['input']
    add_second_in = test_data.get('second_input')
    add_out = test_data['output']

    test_kernel_from_file(
        cl_source,
        kernel_name=kernel_name,
        inputs=[add_first_in.flatten(),
                add_second_in.flatten()] if two_inputs else
                [add_first_in.flatten()],
        expected_outputs=[add_out.flatten()],
        global_size=add_first_in.flatten().shape,
        local_size=(1, ))


test_add()
