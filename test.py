from pymilvus import model, connections
from pymilvus import MilvusClient
import pandas as pd
from pymilvus.orm import utility
from tqdm import tqdm
import logging
from dotenv import load_dotenv
load_dotenv()
import torch

connections.connect("default", host="localhost", port="19530")

# 查询所有 Collection 名称
collections = utility.list_collections()
print("All Collections:", collections)