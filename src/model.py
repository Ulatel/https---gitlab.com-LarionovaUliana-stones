import logging
import json
import fire

from utils.log import logger_init
from utils import const
from utils.config import Config
from utils.data_prepare import DataPreparer

from models.yolo import YoloModel

logger = logging


class My_Model:
    def __init__(self, config):
        self.config = config
        self.data = DataPreparer(self.config)
        if self.config.model_name == 'yolo':
            self.model = YoloModel(self.config)
        else:
            self.model = None

    def train(self, path):
        self.model.train(path)
        
    def evaluate(self, path):
        self.model.evaluate(path)
        
    def warmup(self, path=const.YOLO_PRETRAINED_PATH):
        self.model.warmup(path)
    
    def predict(self, path='example.jpg'):
        self.model.predict(path)    
    
    def demo(self, path='data/train.mp4'):
        self.model.demo(path)
        
  
def init_config(model_name=const.MODEL_NAME):
    config = Config(
        img_path=const.IMAGE_DIR,
        annotation_path=const.ANNOTATION_DIR,
        log_path=const.LOG_PATH,
        log_eval_path=const.LOG_PATH,
        need_training=const.NEED_TRAIN,
        model_name=model_name,
        model_path=const.MODELS_PATH,
        task=const.TASK,
        logger=logger_init(),
    )
    return config


config = init_config()

model = My_Model(config)
model.warmup()
#model.predict()
#model.demo(path='data/clodding_test_ololo.mp4')


class CLI():
    def train(self, data_path=const.MODELS_PATH):
        try:
            model.train(data_path)
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )
        
    def evaluate(self, model_path = const.MODELS_PATH):
        try:
            model.evaluate(model_path)
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )

    def warmup(self, model_path=const.YOLO_PRETRAINED_PATH):
        try:
            model.warmup(model_path)
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )
            
    def predict(self, path='example.jpg'):
        try:
            top_indices, top_scores = model.predict()
            return json.dumps({})
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )
            
    def predict_video(self, path='data/clodding_test_ololo.mp4'):
        try:
            model.demo(path=path)
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )

if __name__ == '__main__':
    fire.Fire(CLI)
