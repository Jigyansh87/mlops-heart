import pandas as pd
import os

def test_data_file_exists():
    assert os.path.exists("data/heart_cleaned.csv")

def test_target_column_exists():
    df = pd.read_csv("data/heart_cleaned.csv")
    assert "target" in df.columns

def test_no_missing_values():
    df = pd.read_csv("data/heart_cleaned.csv")
    assert df.isnull().sum().sum() == 0
