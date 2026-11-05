import numpy as np
import pandas as pd

FEATURES = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
MODEL_COLUMNS = FEATURES + ["temp_diff", "power_w", "wear_torque"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-based features. Used by BOTH training and the API."""
    d = df.copy()
    d["temp_diff"] = d["process_temp"] - d["air_temp"]
    d["power_w"] = d["torque"] * d["rpm"] * 2 * np.pi / 60  # watts
    d["wear_torque"] = d["tool_wear"] * d["torque"]
    return d