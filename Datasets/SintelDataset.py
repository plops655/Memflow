import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class SintelDataset(Dataset):

    def __init__(self, img_dir, ):
