import shutil
from pathlib import Path
import kagglehub

def download_data(output_path: Path = Path("data/raw/winequality-red.csv")) -> Path:
    """Downloads the red wine quality dataset from Kaggle via kagglehub."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download dataset from Kaggle
    dataset_dir = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
    
    # Locate the downloaded CSV file
    downloaded_files = list(Path(dataset_dir).glob("*.csv"))
    if not downloaded_files:
        raise FileNotFoundError("No CSV dataset found in downloaded Kaggle hub directory.")
    
    # Move/Copy to data/raw/winequality-red.csv
    src_csv = downloaded_files[0]
    shutil.copy(src_csv, output_path)
    print(f"✓ Dataset saved to {output_path}")
    return output_path

if __name__ == "__main__":
    download_data()
