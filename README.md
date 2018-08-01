# Master of Kernel Testing

**MOKT** is a library for data-driven testing of OpenCL kernels. It obtains valid
inputs and outputs from TensorFlow models — this way, you can easily get test data for a wide
variety of machine learning operations, ranging from primitives such as
[ReLU](https://github.com/band-of-four/master-of-kernel-testing/blob/master/examples/relu.py)
and [element-wise addition](https://github.com/band-of-four/master-of-kernel-testing/blob/master/examples/add.py)
to whole subgraphs, e.g. ResNet's bottleneck blocks.

## Installation

Install using pip:

```
pip install mokt
```

Note that only Python 3 is supported.

## Usage

Check out [examples](https://github.com/band-of-four/master-of-kernel-testing/tree/master/examples)
to see MOKT in action.

### Data extraction

It is recommended that you read the
[data extraction design note](https://github.com/band-of-four/master-of-kernel-testing/blob/master/design_notes/Extracting%20Test%20Data%20from%20TensorFlow%20Models.ipynb) to get familiar with the way MOKT interacts with TensorFlow.

The high-level data API is used as follows:

```python
@TestData(
    tf_checkpoint_dir='/path/to/checkpoint/dir',
    tf_values={'input': 'operation/name:0', 'output': 'another/op:0'})
def my_test_func(test_data):
    print(type(test_data['input'])) # <class 'numpy.ndarray'>

my_test_func()
```

Choosing the correct nodes for your tests is easier with [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard),
which visualizes the computational graph and shows helpful info, such as tensor shapes, operation names, etc.

### Running OpenCL kernels

Execution is performed in a [TestEnvironment](https://github.com/band-of-four/master-of-kernel-testing/blob/master/mokt/test_environment.py),
which conveniently wraps host state and handles data conversion (read the class documentation for more information).

You may of course choose to write your own specialized implementation and use this library for data extraction only.

### Using universal test runner

```python
python3 com.py --some_keys
```

Keys:
--input/--first_input, --second_input*, --output, --cl_source*, --kernel_name*, --file*

'*' - optionally

If you do not specify 'cl_source' and 'kernel name' then they will be identified automatically by the output operation name

If your output ends with '.../Relu:0', then 'cl_source' will be 'relu.cl' and 'kernel_name' - 'relu'
