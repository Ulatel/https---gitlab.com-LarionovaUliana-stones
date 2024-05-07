from functools import partial
from ultralytics import YOLO
from utils import const
import darknet
import os
import datetime

from ultralytics import YOLO
import numpy as np
import cv2

class YoloModel:
    def __init__(self, config):
        #  YOLO model initialization
        self.config = config
        self.yolo = YOLO('best.pt', task=config.task)
        self.config.logger.info(f"YOLO inited: model_path: {'best.pt'}; task: {config.task}")

    def train(self, images=640, epochs=const.EPOCHS, batch=const.BATCH):
        self.config.logger.info("YOLO started train")
        self.yolo.train(data=const.MODELS_PATH, imgsz=images, epochs=epochs, batch=batch)
        #self._save_model()
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
    
    def _save_model(self):

        cfg_file = "yolov3.cfg"
        weights_file = "yolov3.weights"
        darknet.save_weights(self.model, "saved_weights.weights")

        # Сохранение конфигурации модели
        cfg_buffer = darknet.model_to_json(self.model)
        with open("saved_config.cfg", 'w') as cfg_file:
            cfg_file.write(cfg_buffer.decode("utf-8"))
            
        self.config.logger.info('started _save_model method')
        
        if not os.path.exists(const.MODELS_FOLDER_PATH):
            os.mkdir(const.MODELS_FOLDER_PATH)
            
        new_path = str('_'.join(['YOLO', str(datetime.datetime.now().timestamp()).split(".")[0]]))+'.pt'
        os.rename('runs\detect\train', os.path.join(const.MODELS_FOLDER_PATH, new_path))
        
        self.config.logger.info('_save_model method successfully executed')
        # TODO: update this

    def warmup(self, model_path=const.YOLO_PRETRAINED_PATH) -> None:
        """Method that wormup a model.

        Parameters:
        - model_path: Optional[str], path to model
        """

        self.config.logger.info('Started load_model method')

        self.model = YOLO('best.pt', task=const.TASK)
        self.config.logger.info(f"loaded model: {const.YOLO_PRETRAINED_PATH}")

        self.config.logger.info('load_model method successfully executed')
             
    def demo(self, path='clodding_train.avi', track=False):
        pass