from dataclasses import dataclass

@dataclass
class Config:
    img_path: str
    annotation_path: str
    log_path: str
    log_eval_path: str
    need_training: bool
    model_name: str
    model_path: str
    task: str
    logger: None
