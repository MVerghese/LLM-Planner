from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import base64

def load_images(image_paths):
    """Load images from a list of file paths."""
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    return images


def encode_image(img):
    """Encode image to base64 string"""
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        numpy_img = np.array(img)
    else:
        numpy_img = img
        
    # Convert RGB to BGR for OpenCV
    if len(numpy_img.shape) == 3 and numpy_img.shape[2] == 3:
        numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        
    _, JPEG = cv2.imencode('.jpeg', numpy_img)
    return base64.b64encode(JPEG).decode('utf-8')

def precompute_dataset_embeddings(knn_set: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Precompute the embeddings for the dataset using the specified model.

    Args:
        knn_set (pd.DataFrame): The dataset containing the text to embed.
        model_name (str): The name of the model to use for embedding.

    Returns:
        pd.DataFrame: The original dataset with an additional column for embeddings.
    """
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(knn_set["task_instr"].tolist(), show_progress_bar=True)
    # Cast embeddings to float
    embeddings = embeddings.astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    knn_set["embeddings"] = embeddings.tolist()
    return knn_set

def update_knn_set_with_embeddings(knn_set_path: str, model_name: str) -> pd.DataFrame:
    """
    Load the KNN dataset, compute embeddings, and save the updated dataset.

    Args:
        knn_set_path (str): The file path to the KNN dataset.
        model_name (str): The name of the model to use for embedding.

    Returns:
        pd.DataFrame: The updated KNN dataset with embeddings.
    """
    knn_set = pd.read_pickle(knn_set_path)
    knn_set = precompute_dataset_embeddings(knn_set, model_name)
    knn_set.to_pickle(knn_set_path)
    return knn_set

def explore_knn_set(knn_set_path: str) -> None:
    """
    Explore the KNN dataset by printing basic statistics and a sample of the data.

    Args:
        knn_set (pd.DataFrame): The KNN dataset to explore.
    """
    knn_set = pd.read_pickle(knn_set_path)
    print("KNN Headers:")
    print(knn_set.columns.tolist())
    print("KNN Set Overview:")
    print(knn_set.describe())
    print("\nSample Data:")
    print(knn_set.sample(5))
    import pdb; pdb.set_trace()

def main():
    knn_set_path = "src/knn_set.pkl"
    knn_set_path = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/prompt_iteration/knn_sets/knn_set_iteration_3.pkl"
    model_name = "paraphrase-MiniLM-L6-v2"

    # updated_knn_set = update_knn_set_with_embeddings(knn_set_path, model_name)
    # print("Updated KNN Set:")
    # print(updated_knn_set)
    explore_knn_set(knn_set_path)

if __name__ == "__main__":
    main()