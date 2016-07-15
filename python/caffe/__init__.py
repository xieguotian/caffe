from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver, RMSPropSolver, AdaDeltaSolver, AdamSolver, SetDataLayerSource
from ._caffe import set_mode_cpu, set_mode_gpu, set_device, Layer, get_solver, layer_type_list, set_random_seed, TrainManager
from ._caffe import __version__
from .proto.caffe_pb2 import TRAIN, TEST
from .classifier import Classifier, Classifier_parallel, Classifier_dense
from .detector import Detector
from . import io
from .net_spec import layers, params, NetSpec, to_proto
