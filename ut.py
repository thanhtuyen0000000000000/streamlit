# File: utils.py
import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import uuid
from tqdm import tqdm

# Thiết lập Qdrant client
def setup_qdrant():
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name="mycollection",
        vectors_config={
            "image": models.VectorParams(
                size=512,
                distance=models.Distance.COSINE
            )
        }
    )
    return client

# Load dữ liệu và upsert lên Qdrant (chỉ thực hiện 1 lần)
def load_data_and_upsert(client):
    df = pd.read_csv(r"D:\UIT\aic\combined_data.csv")  # Đọc dữ liệu từ CSV
    CLIP_V001 = np.load(r"D:\UIT\aic\combined_features_fix.npy")  # Load CLIP features từ file .npy
    
    test_dataset = Dataset.from_pandas(df)
    points = []
    for embedding, image_path, idx, pts_time in tqdm(zip(CLIP_V001, df["Image_Path"], test_dataset["frame_idx"], test_dataset["Name_Vid"]), total=len(df)):
        payload = {"Image Name": idx, "Name_Vid": pts_time, "image": image_path}
        image_embedding = embedding.tolist()
        point_id = str(uuid.uuid4())
        points.append(PointStruct(id=point_id, vector={"image": image_embedding}, payload=payload))

    client.upsert(collection_name="mycollection", points=points, wait=True)

# Load mô hình CLIP và bộ tiền xử lý
def load_clip_model():
    import clip
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess
