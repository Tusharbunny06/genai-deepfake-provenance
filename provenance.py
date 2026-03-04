import hashlib
import time
import json
import os

LEDGER_FILE = "provenance_chain.json"


def sha256(data):
    return hashlib.sha256(data.encode()).hexdigest()


def generate_video_hash(video_bytes):
    return hashlib.sha256(video_bytes).hexdigest()


def load_chain():

    if not os.path.exists(LEDGER_FILE):
        return []

    with open(LEDGER_FILE, "r") as f:
        return json.load(f)


def save_chain(chain):

    with open(LEDGER_FILE, "w") as f:
        json.dump(chain, f, indent=4)


def add_video_record(video_hash):

    chain = load_chain()

    index = len(chain)
    timestamp = str(time.time())

    if index == 0:
        prev_hash = "GENESIS"
    else:
        prev_hash = chain[-1]["block_hash"]

    block_data = str(index) + timestamp + video_hash + prev_hash
    block_hash = sha256(block_data)

    block = {
        "index": index,
        "timestamp": timestamp,
        "video_hash": video_hash,
        "previous_hash": prev_hash,
        "block_hash": block_hash
    }

    chain.append(block)
    save_chain(chain)

    return block_hash


def verify_video(video_hash):

    chain = load_chain()

    for block in chain:
        if block["video_hash"] == video_hash:
            return True

    return False