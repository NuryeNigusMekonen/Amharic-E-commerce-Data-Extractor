import os
import time
import logging
from datetime import datetime
import pandas as pd
from telethon import TelegramClient
# CONFIG 
API_ID = 21248507  #  
API_HASH = '11dbaaa78d13696390b000ce2a029b2d'
SESSION_NAME = 'ecommerce_scraper'
CHANNELS = [
    'https://t.me/ZemenExpress',
    'https://t.me/Fashiontera',
    'https://t.me/nevacomputer',
    'https://t.me/ethio_brand_collection',
    'https://t.me/Shewabrand'
]

BASE_DIR = '..'  # Since notebook is in /notebooks/, go up one level
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MEDIA_DIR = os.path.join(RAW_DIR, 'media')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
# LOGGING 
log_file = os.path.join(LOG_DIR, f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    filename=log_file,
    filemode='w',
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
#SCRAPER
async def scrape():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()
    logging.info(" Connected to Telegram API.")

    all_msgs = []

    for ch in CHANNELS:
        try:
            entity = await client.get_entity(ch)
            async for msg in client.iter_messages(entity, limit=100):
                msg_data = {
                    'channel': ch,
                    'sender_id': msg.sender_id,
                    'timestamp': msg.date.isoformat(),
                    'message': msg.message,
                    'views': msg.views,
                    'media_file': None
                }

                if msg.media:
                    file_name = await msg.download_media(file=os.path.join(MEDIA_DIR, f"{msg.id}"))
                    msg_data['media_file'] = file_name
                    logging.info(f"Downloaded media {file_name} from {ch}")

                all_msgs.append(msg_data)

            logging.info(f" Finished scraping {ch}. Messages so far: {len(all_msgs)}")

        except Exception as e:
            logging.error(f" Error scraping {ch}: {e}")
        time.sleep(2)

    if all_msgs:
        df = pd.DataFrame(all_msgs)
        out_file = os.path.join(RAW_DIR, f"raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(out_file, index=False)
        logging.info(f" Saved raw data to {out_file}")
        return out_file
    else:
        logging.warning("⚠ No messages scraped.")
        return None
