from dataclasses import dataclass, field
from typing import Tuple, Any

import cv2
import yaml
from pydantic import BaseModel, Field


class CountLine(BaseModel):
    start_point: Tuple = (350, 0)  # from top left (x, y)
    end_point: Tuple = (350, 1080)
    color: Tuple = (0, 0, 255)  # BGR
    thickness: int = 2


# This is for the object moving direction visualize
class DirVisualize(BaseModel):
    thickness: int = 1
    radius: int = 2
    color: Tuple = (0, 255, 0)  # BGR


# This is for object circle line
class CountCircle(BaseModel):
    color: Tuple = (0, 0, 255)  # BGR
    thickness: int = 2
    radius: int = 270
    center: Tuple = (900, 270)


# This is for text
class Text(BaseModel):
    font: Any = cv2.FONT_HERSHEY_SIMPLEX
    org: Tuple = (50, 50)  # start position (from top left)
    fontScale: int = 1
    color: Tuple = (0, 255, 0)  # BGR
    thickness: int = 2


class Config(BaseModel):
    count_line: CountLine = CountLine()
    dir_visualize: DirVisualize = DirVisualize()
    count_circle: CountCircle = CountCircle()
    text: Text = Text()


with open('config.yaml', 'r') as file:
    data = yaml.safe_load(file)

config = Config(**data)
print(config.dir_visualize.color)
