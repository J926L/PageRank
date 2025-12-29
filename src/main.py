import os
import sys
from pathlib import Path

# CUDA 显存限制配置 (RTX 3060 6GB constraint)
# Even if not using torch, this satisfies the P5 protocol.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Limit memory growth if using tensorflow/torch (optional but good practice)
# import torch; if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(0.8, 0)

from app import run_pagerank

def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "input"
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} not found.")
        sys.exit(1)
    
    print("Starting PageRank analysis...")
    run_pagerank(data_dir)

if __name__ == "__main__":
    main()
