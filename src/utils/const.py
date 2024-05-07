# Data path
IMAGE_DIR = 'data/clodding_train'
ANNOTATION_DIR = 'data/yolo'
LOG_PATH = 'data/log_file.log'
MODELS_FOLDER_PATH = 'models'

# Model configuration
MODEL_NAME = 'yolo'
NEED_TRAIN = True
EPOCHS = 1
BATCH = 4
MODELS_PATH = 'datasets/data.yaml'
YOLO_PATH = MODELS_FOLDER_PATH+'/yolo'
YOLO_PRETRAINED_PATH = 'runs/segment/train11/weights/best.pt'
TASK = 'segment'

# Augmentation parameters
N_SHAPEN = 1
ALPHA_CONSTRAT = 1.7
