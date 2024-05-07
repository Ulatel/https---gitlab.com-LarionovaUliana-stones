# Data path
IMAGE_DIR = 'src/data/clodding_train'
ANNOTATION_DIR = 'src/data/yolo'
LOG_PATH = 'src/data/log_file.log'
MODELS_FOLDER_PATH = 'src/models'

# Model configuration
MODEL_NAME = 'yolo'
NEED_TRAIN = True
EPOCHS = 1
BATCH = 4
MODELS_PATH = 'src/datasets/data.yaml'
YOLO_PATH = MODELS_FOLDER_PATH+'/yolo'
YOLO_PRETRAINED_PATH = 'best.pt'
TASK = 'segment'

# Augmentation parameters
N_SHAPEN = 1
ALPHA_CONSTRAT = 1.7
