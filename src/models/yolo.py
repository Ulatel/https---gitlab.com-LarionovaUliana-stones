
from ultralytics import YOLO
from src.logger import logger

class YoloModel:
    def __init__(self, config):
        #  YOLO model initialization
        self.config = config
        self.yolo = YOLO(config.model_path, task=config.task)
        logger.info(f"YOLO inited: model_path: {config.model_path}; task: {config.task}")


    def train(self, train_data, validation_data):
        logger.info(f"YOLO started train")
        self.estimator.train(data=data, imgsz=imgsz, epochs=epochs, batch=batch)
        logger.info(f"YOLO finished train")

    def evaluate(self, evaluation_data):
        # Оценка модели YOLO
        pass

    def predict(self, input_data):
        # Предсказание с помощью модели YOLO
        pass
