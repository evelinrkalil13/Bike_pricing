import os
import pickle
 
 
def load_model(model_path: str = "models/model.pkl"):
    """
    Load a model from a pickle file.
 
    Parameters:
    model_name (str): The name of the model to load.
 
    Returns:
    object: The loaded model.
    """
    try:
        with open(model_path, "rb") as file:
            pipeline = pickle.load(file)
        print(f"Model loaded from {model_path}")
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
