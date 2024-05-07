import logging
import numpy as np
import cv2
import random
import os
import json
import fire

from utils.log import logger_init
from utils import const
from utils.config import Config

from models.yolo import YoloModel

logger = logging


class DataPreparer:
    """Класс для подготовки данных."""

    def __init__(
        self, config,
    ) -> None:
        self.config = config
        self.train_img = None
        self.test_img = None
        self.train_annot = None
        self.test_annot = None
        self.sharpen_filter = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]])
        self.logger = logger  

    def _apply_sharpen_filter(self, img, n=const.N_SHAPEN):
        self.logger.info('started sharpening of image')  
        if img is None or n <= 0:
            raise ValueError("Invalid input parameters")

        sharpened_img = img.copy()
        for _ in range(n):
            sharpened_img = cv2.filter2D(sharpened_img, -1, self.sharpen_filter)

        self.logger.info('successfully sharpen image')  
        return sharpened_img

    def _change_contrast(self, img, alpha=const.ALPHA_CONSTRAT):
        self.logger.info('started changing contrast of image')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = img * alpha
        img = np.clip(img, 0, 255)
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.logger.info('contrast successfully changed')
        return img
    
    def _read_yolo_annotation(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        annotations = []
        for line in lines:
            values = line.split(' ')
            class_id = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])
            annotations.append((class_id, x_center, y_center, width, height))
        return np.array(annotations)
    
    def _split_data(self, images, annotations, test_percentage=0.2):
        data = list(zip(images, annotations))
        random.shuffle(data)
        split_index = int(len(data) * test_percentage)
        test_data = data[:split_index]
        train_data = data[split_index:]
        test_images, test_annotations = zip(*test_data)
        train_images, train_annotations = zip(*train_data)
        return train_images, train_annotations, test_images, test_annotations
    
    def _data_init(self):
        image_data = []
        annotation_data = []

        for filename in os.listdir(const.IMAGE_DIR):
            img_path = os.path.join(const.IMAGE_DIR, filename)
            annotation_path = os.path.join(const.ANNOTATION_DIR, os.path.splitext(filename)[0] + '.txt')
            if os.path.isfile(img_path) and os.path.isfile(annotation_path):
                image = self._apply_sharpen_filter(cv2.imread(img_path))
                image_data.append(image)
                image_data.append(self._change_contrast(image))
                annotations = self._read_yolo_annotation(annotation_path)
                annotation_data.append(annotations)
                annotation_data.append(annotations)
        
        self.logger.info(f"Images count: {len(image_data)}")
        self.logger.info(f"Annotations count: {len(annotation_data)}")
        
        return image_data, annotation_data

    def prepare_data(self):
        self.logger.info("Start preparing data for prediction")
        image_data, annotation_data = self._data_init()
        self.train_img, self.test_img, self.train_annot, self.test_annot = self._split_data(image_data, annotation_data)
        self.logger.info("End preparing data for prediction")


class Shower:
    def __init__(self, config):
        self.config = config
        

class My_Model:
    def __init__(self, config):
        self.config = config
        self.shower = Shower(self.config)
        self.data = DataPreparer(self.config)
        if self.config.model_name == 'yolo':
            self.model = YoloModel(self.config)
        else:
            self.model = None

    def train(self):
        # self.data.prepare_data()
        self.model.train()
        
    def evaluate(self):
        self.model.evaluate()
        
    def warmup(self, path=const.YOLO_PRETRAINED_PATH):
        self.model.warmup(path)
    
    def predict(self, path='example.jpg'):
        self.model.predict(path)    
    
    def demo(self):
        self.model.demo()
        
  
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
model.predict()
model.demo()


class CLI():
    def train(self):
        try:
            model.train()
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )
        
    def evaluate(self):
        try:
            model.evaluate()
        except Exception as ex:
            raise Exception(
                detail=str(ex),
                status_code=ex.code if hasattr(object, 'code') else 500
            )

    def warmup(self, path):
        try:
            model.warmup(path)
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
            
    def predict_video(self, path='clodding_train.avi'):
        pass


"""if __name__ == '__main__':
    fire.Fire(CLI)
"""