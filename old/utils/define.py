#ROOT_PATH = '/path/to/repo/3DSSG/'
#DATA_PATH = '/path/to/3RScan/'
#SCANNET_DATA_PATH = '/path/to/scannet' 
ROOT_PATH = '/home/sc/research/PersistentSLAM/python/3DSSG/'
DATA_PATH = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'
SCANNET_DATA_PATH = '/media/sc/space1/dataset/scannet/scans/'
SCANNET_SPLIT_TRAIN = '/path/to/scannet/Tasks/Benchmarkscannetv2_train.txt'
SCANNET_SPLIT_VAL = '/path/to/scannet/Tasks/Benchmark/scannetv2_val.txt'

FILE_PATH = ROOT_PATH+'files/'
Scan3RJson_PATH = FILE_PATH+'3RScan.json'
LABEL_MAPPING_FILE = FILE_PATH+'3RScan.v2 Semantic Classes - Mapping.csv'
CLASS160_FILE = FILE_PATH+'classes160.txt'
RELATIONSHIP27_FILE = FILE_PATH+'relationships.txt'

# 3RScan file names
LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.align.annotated.v2.ply'
SEMSEG_FILE_NAME = 'semseg.v2.json'
MTL_NAME = 'mesh.refined.mtl'
OBJ_NAME = 'mesh.refined.obj'
TEXTURE_NAME = 'mesh.refined_0.png'

# ScanNet file names
SCANNET_SEG_SUBFIX = '_vh_clean_2.0.010000.segs.json'
SCANNET_AGGRE_SUBFIX = '.aggregation.json'
SCANNET_PLY_SUBFIX = '_vh_clean_2.labels.ply'


NAME_SAME_PART = 'same part'
