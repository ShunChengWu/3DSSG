from . import node_encoder
from . import MVCNN
from . import GVCNN
from . import MHBNN
# from . import CVR
from . import mean
from . import mvgmu
from . import ROIExtractor
node_encoder_list = {
    'basic': node_encoder.NodeEncoder,
    'sgfn': node_encoder.NodeEncoder_SGFN,
    'vgg16': node_encoder.NodeEncoderVGG16,
    'resnet18': node_encoder.NodeEncoderRes18,
    'mvcnn': MVCNN.MVCNN,
    'gvcnn': GVCNN.GVCNN,
    'svcnn': GVCNN.SVCNN,
    'mhbnn': MHBNN.MHBNN,
    # 'cvr': CVR.CVR,
    'mean': mean.MeanMV,
    'gmu': mvgmu.MVGMU,
    'roi_extractor': ROIExtractor.ROI_EXTRACTOR,
}
