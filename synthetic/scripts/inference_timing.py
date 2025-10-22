import torch
import time
import numpy as np
import argparse
import os
import glob
from torch.utils.data import DataLoader
from CHiLD.modules.instantaneous import InstantaneousProcess
from CHiLD.datasets.sim_dataset import StationaryDataset
from CHiLD.tools.utils import load_yaml
import warnings
warnings.filterwarnings('ignore')

def load_trained_model(config_name, checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint"""
    # Load config
    config_path = f'../CHiLD/configs/{config_name}.yaml'
    cfg = load_yaml(config_path)
    
    # Create model
    model = InstantaneousProcess(
        input_dim=cfg['VAE']['INPUT_DIM'],
        z_dim=cfg['VAE']['LATENT_DIM'], 
        z_dim_fix=cfg['VAE']['LATENT_DIM_FIX'],
        z_dim_change=cfg['VAE']['LATENT_DIM_CHANGE'],
        lag=cfg['VAE']['LAG'],
        nclass=cfg['VAE']['NCLASS'],
        hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
        embedding_dim=cfg['VAE']['EMBED_DIM'],
        lr=cfg['VAE']['LR'],
        beta=cfg['VAE']['BETA'],
        gamma=cfg['VAE']['GAMMA'],
        theta=cfg['VAE']['THETA'],
        decoder_dist=cfg['VAE']['DEC']['DIST'],
        correlation=cfg['MCC']['CORR'],
        enable_flexible_sparsity=cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'],
        w_hist=cfg['VAE']['FLEXIBLE_SPARTSITY']['HIST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
        w_inst=cfg['VAE']['FLEXIBLE_SPARTSITY']['INST'] if cfg['VAE']['FLEXIBLE_SPARTSITY']['ENABLE'] else None,
        z_dim_list=cfg['VAE']['Z_DIM_LIST'],
        n_mea=cfg['VAE']['N_MEA'],
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, cfg

def measure_inference_time(model, data_loader, device='cuda', num_batches=10, warmup_batches=3):
    """Measure inference time for model evaluation"""
    model.eval()
    
    # Warmup runs
    print(f"Running {warmup_batches} warmup batches...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= warmup_batches:
                break
            x, y = batch
            x, y = x.to(device), y.to(device)
            _ = model(x, y)
    
    # Actual timing
    print(f"Measuring inference time over {num_batches} batches...")
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches + warmup_batches:
                break
            if i < warmup_batches:
                continue
                
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # Time the forward pass
            torch.cuda.synchronize()  # Ensure all operations are complete
            start_time = time.time()
            
            output = model(x, y)
            
            torch.cuda.synchronize()  # Ensure all operations are complete
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
            
            print(f"Batch {i-warmup_batches+1}/{num_batches}: {batch_time:.4f}s")
    
    return times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['N', 'O'], help='Dataset to test (N or O)')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint path (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num_batches', type=int, default=10, help='Number of batches to time')
    parser.add_argument('--warmup_batches', type=int, default=3, help='Number of warmup batches')
    
    args = parser.parse_args()
    
    # Find a checkpoint if not specified
    if args.checkpoint is None:
        # Look for recent checkpoints for this dataset
        checkpoint_pattern = f"./CHILD/*/checkpoints/*.ckpt"
        checkpoints = glob.glob(checkpoint_pattern)
        if not checkpoints:
            print(f"No checkpoints found for dataset {args.dataset}")
            return
        
        # Use the most recent checkpoint
        args.checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Using checkpoint: {args.checkpoint}")
    
    # Load model
    print(f"Loading model for dataset {args.dataset}...")
    model, cfg = load_trained_model(args.dataset, args.checkpoint, args.device)
    
    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    data = StationaryDataset(args.dataset)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Measure inference time
    times = measure_inference_time(model, data_loader, args.device, args.num_batches, args.warmup_batches)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate per-sample time
    per_sample_time = mean_time / args.batch_size
    
    print("\n" + "="*50)
    print(f"INFERENCE TIME RESULTS - Dataset {args.dataset}")
    print("="*50)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Device: {args.device}")
    print(f"Model config: {args.dataset}.yaml")
    print(f"Checkpoint: {os.path.basename(args.checkpoint)}")
    print("-"*50)
    print(f"Mean time per batch: {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"Min time per batch:  {min_time:.4f} seconds")
    print(f"Max time per batch:  {max_time:.4f} seconds")
    print(f"Mean time per sample: {per_sample_time:.6f} seconds")
    print(f"Throughput: {1/per_sample_time:.2f} samples/second")
    print("="*50)

if __name__ == "__main__":
    main()