import os
import requests

# Constants and Configuration
RAW_DATA_DIR = os.path.join("data", "raw")
URL_FREQ = "https://www.openml.org/data/get_csv/20649148/freMTPL2freq.arff"
URL_SEV = "https://www.openml.org/data/get_csv/20649149/freMTPL2sev.arff"

def setup_directories() -> None:
    """
    Ensure the raw data directory exists before attempting to write files.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"[INFO] Verified directory: {RAW_DATA_DIR}")

def download_file(url: str, filename: str) -> None:
    """
    Download a file from a specified URL in chunks to optimize memory usage.
    
    Args:
        url (str): The source URL of the dataset.
        filename (str): The target filename to save in the raw data directory.
    """
    target_path = os.path.join(RAW_DATA_DIR, filename)
    print(f"[INFO] Starting download from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(target_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    
        print(f"[INFO] Successfully downloaded: {target_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download {filename}. Reason: {e}")

def main() -> None:
    """
    Main execution flow for the data ingestion process.
    """
    print("[INFO] Starting Data Ingestion Pipeline...")
    setup_directories()
    
    download_file(URL_FREQ, "freMTPL2freq.csv")
    download_file(URL_SEV, "freMTPL2sev.csv")
    
    print("[INFO] Data ingestion completed successfully.")

if __name__ == "__main__":
    main()