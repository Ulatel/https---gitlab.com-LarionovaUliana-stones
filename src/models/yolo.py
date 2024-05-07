from functools import partial
from ultralytics import YOLO
from utils import const
#import darknet
import os
import datetime

from ultralytics import YOLO
import numpy as np
import cv2


class YoloModel:
    def __init__(self, config):
        #  YOLO model initialization
        self.config = config
        self.yolo = YOLO('runs/segment/train11/weights/best.pt', task=config.task)
        self.config.logger.info(f"YOLO inited: model_path: {'best.pt'}; task: {config.task}")

    def train(self, images=640, epochs=const.EPOCHS, batch=const.BATCH):
        self.config.logger.info("YOLO started train")
        self.yolo.train(data=const.MODELS_PATH, imgsz=images, epochs=epochs, batch=batch)
        self.config.logger.info("YOLO finished train")

    def evaluate(self, images=640):
        # Оценка модели YOLO
        self.config.logger.info("YOLO started validation")
        results = self.yolo.val(data=const.MODELS_PATH, imgsz=images)
        output = {
            'mAP50': results.results_dict['metrics/mAP50(M)'],
            'precision': results.results_dict['metrics/precision(B)'],
            'recall': results.results_dict['metrics/recall(B)'],
            'f1': results.box.f1[0]
        }
        self.config.logger.info("YOLO finished validation")
        return output
      
    def predict(self, path: str, task=const.TASK, save=True, save_txt=True, stream=True):
        # Предсказание с помощью модели YOLO\

        self.config.logger.info(f"YOLO started prediction: {path}")
        generators = self.yolo.predict(source=path, task=task, save=save, save_txt=save_txt, stream=stream)
        print(generators)
        for _ in generators:
            print(_)
        self.config.logger.info(f"YOLO predicted: {path}")
    
    def warmup(self, model_path=const.YOLO_PRETRAINED_PATH) -> None:
        """Method that wormup a model.

        Parameters:
        - model_path: Optional[str], path to model
        """

        self.config.logger.info('Started load_model method')

        self.model = YOLO(model_path, task=const.TASK)
        self.config.logger.info(f"loaded model: {model_path}")

        self.config.logger.info('load_model method successfully executed')
             
    def demo(self, path='clodding_train.avi', track=False):
        pass