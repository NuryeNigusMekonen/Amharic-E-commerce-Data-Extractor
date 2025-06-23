import os
import pandas as pd
import logging
from datetime import datetime
from sacremoses import MosesTokenizer
import regex as re  # Use regex for unicode support

# PATHS 
BASE_DIR = '..'  
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'merged_cleaned.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# LOGGING 
log_file = os.path.join(LOG_DIR, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
# TOKENIZER 
tokenizer = MosesTokenizer(lang='am')
# CLEANING FUNCTION 
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove Telegram handles
    text = re.sub(r'@\S+', '', text)

    # Remove emojis/symbols/non-Amharic
    text = re.sub(r'[^\p{IsEthiopic}\w\s፡።,.-]', '', text)

    # Remove excessive dots
    text = re.sub(r'\.{2,}', '.', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
# PREPROCESS FUNCTION
def preprocess(file_path=INPUT_FILE):
    logging.info(f" Loading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f" File not found: {file_path}")
        return None
    if 'message' not in df.columns:
        logging.error(" 'message' column not found in the input data.")
        return None
    # Clean and tokenize
    df['clean_message'] = df['message'].apply(clean_text)
    df['tokens'] = df['clean_message'].apply(lambda x: tokenizer.tokenize(x))
    # Output path
    out_file = os.path.join(PROCESSED_DIR, f"processed_{os.path.basename(file_path)}")
    df.to_csv(out_file, index=False)
    logging.info(f" Saved processed data to: {out_file}")
    return out_file
# Optional: allow running directly
if __name__ == "__main__":
    preprocess()
