"""Load the public AlphaChu-v1 policy for the CUDA training opponent."""

import hashlib
import os
import tempfile
import urllib.request

import numpy as np
import onnx
from onnx import numpy_helper


MODEL_REVISION = '911b5f0e47b494552ac436bc1a4d394e445428d8'
MODEL_URL = (
    'https://huggingface.co/alphachu-volleyball/alphachu-v1/resolve/'
    + MODEL_REVISION + '/model.onnx'
)
MODEL_SHA256 = '721d5cfb9fd3e3bec9a73345c7b47696b95ec6d98d48db434a47f0d09d038aee'
MODEL_CARD_URL = 'https://huggingface.co/alphachu-volleyball/alphachu-v1'
MODEL_SIDE = 'player_1'
MODEL_ENV_CONTRACT = 'alphachu-volleyball/pika-zoo:P1_MAP+NormalizeObservation'

# AlphaChu-v1 was trained/exported with alphachu-volleyball/pika-zoo's
# SimplifyAction and NormalizeObservation wrappers. These differ from the
# older helpingstar/pika-zoo wrapper contract used by this project's first
# GPU integration. Keep these values in sync with gpu_physics.py.
OBSERVATION_LOW = np.asarray([
    0, 0, -16, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, -16, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, -20, -30, 0,
], dtype=np.float32)
OBSERVATION_HIGH = np.asarray([
    432, 244, 16, 1, 3, 4, 5, 1, 1, 1, 1, 1, 1,
    432, 244, 16, 1, 3, 4, 5, 1, 1, 1, 1, 1, 1,
    432, 304, 432, 304, 432, 304, 20, 30, 1,
], dtype=np.float32)
P1_ACTION_KEYS = np.asarray([
    [0, 0, 0],   # NOOP
    [0, 0, 1],   # FIRE
    [0, -1, 0],  # UP
    [1, 0, 0],   # RIGHT / toward net
    [-1, 0, 0],  # LEFT / away from net
    [0, 1, 0],   # DOWN
    [1, -1, 0],  # UP+RIGHT
    [-1, -1, 0], # UP+LEFT
    [1, 1, 0],   # DOWN+RIGHT
    [-1, 1, 0],  # DOWN+LEFT
    [0, -1, 1],  # UP+FIRE
    [1, 0, 1],   # RIGHT+FIRE
    [-1, 0, 1],  # LEFT+FIRE
], dtype=np.int32)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path):
    """Download the pinned 30 KB model once and verify its published SHA-256."""
    if os.path.isfile(path):
        actual = _sha256(path)
        if actual != MODEL_SHA256:
            raise ValueError(
                f'AlphaChu model hash mismatch at {path}: {actual}. '
                'Remove or replace that file before training.'
            )
        return path

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='.alphachu-', suffix='.onnx', dir=parent)
    os.close(fd)
    try:
        request = urllib.request.Request(
            MODEL_URL,
            headers={'User-Agent': 'raichu-volleyball-training/1.0'},
        )
        with urllib.request.urlopen(request, timeout=60) as response, open(temp_path, 'wb') as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
        actual = _sha256(temp_path)
        if actual != MODEL_SHA256:
            raise ValueError(
                f'Downloaded AlphaChu model hash mismatch: {actual}; '
                f'expected {MODEL_SHA256}.'
            )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return path


def load_weights(path, array_module):
    """Validate the published ONNX graph and return row-major CUDA-ready arrays."""
    ensure_model(path)
    model = onnx.load(path)
    onnx.checker.check_model(model)

    inputs = [value for value in model.graph.input
              if value.name not in {item.name for item in model.graph.initializer}]
    if len(inputs) != 1 or inputs[0].name != 'obs':
        raise ValueError('Unexpected AlphaChu ONNX input; expected one input named obs.')
    input_shape = [dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim]
    if len(input_shape) != 2 or input_shape[1] != OBSERVATION_LOW.size:
        raise ValueError(f'Unexpected AlphaChu observation shape: {input_shape}.')

    outputs = list(model.graph.output)
    if (len(outputs) != 1 or outputs[0].name != 'action_logits' or
            outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
        raise ValueError('Unexpected AlphaChu ONNX output; expected float action_logits.')
    output_shape = [dim.dim_value for dim in outputs[0].type.tensor_type.shape.dim]
    if len(output_shape) != 2 or output_shape[1] != P1_ACTION_KEYS.shape[0]:
        raise ValueError(f'Unexpected AlphaChu action-logit shape: {output_shape}.')

    nodes = [node for node in model.graph.node if node.op_type != 'Flatten']
    signature = [(node.op_type, tuple(node.input)) for node in nodes]
    expected_ops = ['Gemm', 'Tanh', 'Gemm', 'Tanh', 'Gemm']
    if [node.op_type for node in nodes] != expected_ops:
        raise ValueError(f'Unsupported AlphaChu policy graph: {signature}.')

    for node in (nodes[0], nodes[2], nodes[4]):
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute)
                      for attribute in node.attribute}
        if attributes.get('transB') != 1:
            raise ValueError('AlphaChu linear layers must use transposed ONNX Gemm weights.')

    initializers = {value.name: numpy_helper.to_array(value)
                    for value in model.graph.initializer}
    shapes = {
        'policy_net.0.weight': (64, 35), 'policy_net.0.bias': (64,),
        'policy_net.2.weight': (64, 64), 'policy_net.2.bias': (64,),
        'action_net.weight': (13, 64), 'action_net.bias': (13,),
    }
    arrays = []
    for name, shape in shapes.items():
        value = initializers.get(name)
        if value is None or value.shape != shape or value.dtype != np.float32:
            raise ValueError(f'Unexpected AlphaChu tensor {name}: '
                             f'{None if value is None else (value.shape, value.dtype)}.')
        arrays.append(array_module.asarray(np.ascontiguousarray(value.reshape(-1))))
    return tuple(arrays)
