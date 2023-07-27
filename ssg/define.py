import os
# ROOT_PATH = '/home/sc/research/PersistentSLAM/python/3DSSG/'
# DATA_PATH = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'
# SCANNET_DATA_PATH = '/media/sc/space1/dataset/scannet/scans/'
# SCANNET_SPLIT_TRAIN = '/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_train.txt'
# SCANNET_SPLIT_VAL = '/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt'


# Scan3RJson_PATH = FILE_PATH+'3RScan.json'
# RELATIONSHIP27_FILE = FILE_PATH+'relationships.txt'

# Coco
# PATH_LABEL_COCOSTUFF = './files/cocostuff.txt'

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.align.annotated.v2.ply'
NAME_PATTERN_INSTANCE_IMG = 'frame-{0:06d}.rendered.instances.png'
NAME_PATTERN_LABEL_IMG = 'frame-{0:06d}.rendered.labels.png'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.v2.obj'
TEXTURE_NAME = 'mesh.refined_0.png'
INFO_NAME = '_info.txt'
RELATIONSHIP_NAME = 'relationships.json'

# 3RScan sequence
IMG_FOLDER_NAME = 'sequence'
FRAME_NAME_FORMAT = 'frame-{:06d}'
RGB_NAME_FORMAT = (FRAME_NAME_FORMAT+'.color.jpg')
DEPTH_NAME_FORMAT = (FRAME_NAME_FORMAT+'.depth.pgm')
POSE_NAME_FORMAT = (FRAME_NAME_FORMAT+'.pose.txt')

# ScanNet file names
SCANNET_SEG_SUBFIX = '_vh_clean_2.0.010000.segs.json'
SCANNET_AGGRE_SUBFIX = '.aggregation.json'
SCANNET_PLY_SUBFIX = '_vh_clean_2.labels.ply'

# Defined
NAME_SAME_PART = 'same part'
NAME_NONE = 'none'
NAME_IMAGE_FEAUTRE_FORMAT = 'image_feature_{}_{}'
NAME_FILTERED_KF_INDICES = 'kf_indices'
NAME_FILTERED_OBJ_INDICES = 'obj_indices'

# Processed file realted
TYPE_2DGT = '.2dgt'
PATH_FILE = 'files'
NAME_VIS_GRAPH = 'proposals.h5'
NAME_RELATIONSHIPS = 'relationships.h5'
NAME_ROI_IMAGE = 'roi_images.h5'
PATH_LABEL_MAPPING = os.path.join(
    PATH_FILE, '3RScan.v2 Semantic Classes - Mapping.csv')
PATH_CLASS160_FILE = os.path.join(PATH_FILE, 'classes160.txt')


#
STRUCTURE_LABELS = ['wall', 'floor', 'ceiling']
SUPPORT_TYPE_RELATIONSHIPS = ['supported by', 'attached to',
                              'standing on', 'hanging on', 'connected to', 'part of', 'build in']
