import torch
import time
import numpy as np
import argparse
from torch.utils.data import DataLoader
from CHiLD.modules.instantaneous import InstantaneousProcess
from CHiLD.datasets.sim_dataset import StationaryDataset
from CHiLD.tools.utils import load_yaml
import warnings
warnings.filterwarnings('ignore')

def create_model(config_name, device='cuda'):
    """Create a model from config (without loading weights)"""
    config_path = f'../CHiLD/configs/{config_name}.yaml' 
    cfg = load_yaml(config_path)
    
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
    
    model.to(device)
    model.eval()
    return model, cfg

def measure_inference_time(model, data_loader, device='cuda', num_batches=10, warmup_batches=3):
    """Measure inference time for model evaluation"""
    model.eval()
    
    print(f"Running {warmup_batches} warmup batches...")
    with torch.no_grad():
        batch_iter = iter(data_loader)
        for i in range(warmup_batches):
            try:
                batch = next(batch_iter)
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                else:
                    batch = batch.to(device)
                _ = model(batch)
            except StopIteration:
                batch_iter = iter(data_loader)
                batch = next(batch_iter)
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items()}
                else:
                    batch = batch.to(device)
                _ = model(batch)
    
    print(f"Measuring inference time over {num_batches} batches...")
    times = []
    
    with torch.no_grad():
        batch_iter = iter(data_loader)
        for i in range(num_batches):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(data_loader)
                batch = next(batch_iter)
                
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Time the forward pass
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            output = model(batch)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
            
            print(f"Batch {i+1}/{num_batches}: {batch_time:.4f}s")
    
    return times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['N', 'O'], help='Dataset to test (N or O)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num_batches', type=int, default=20, help='Number of batches to time')
    parser.add_argument('--warmup_batches', type=int, default=5, help='Number of warmup batches')
    
    args = parser.parse_args()
    
    print(f"Creating model for dataset {args.dataset}...")
    model, cfg = create_model(args.dataset, args.device)
    
    print(f"Loading dataset {args.dataset}...")
    data = StationaryDataset(args.dataset)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Model architecture:")
    print(f"  Input dim: {cfg['VAE']['INPUT_DIM']}")
    print(f"  Latent dim: {cfg['VAE']['LATENT_DIM']}")
    print(f"  Z dim list: {cfg['VAE']['Z_DIM_LIST']}")
    print(f"  Hidden dim: {cfg['VAE']['ENC']['HIDDEN_DIM']}")
    
    # Measure inference time
    times = measure_inference_time(model, data_loader, args.device, args.num_batches, args.warmup_batches)
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate per-sample time
    per_sample_time = mean_time / args.batch_size
    
    print("\n" + "="*60)
    print(f"INFERENCE TIME RESULTS - Dataset {args.dataset}")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Device: {args.device}")
    print(f"Model config: {args.dataset}.yaml")
    print("-"*60)
    print(f"Mean time per batch:  {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"Min time per batch:   {min_time:.4f} seconds")
    print(f"Max time per batch:   {max_time:.4f} seconds")
    print(f"Mean time per sample: {per_sample_time*1000:.2f} milliseconds")
    print(f"Throughput:          {1/per_sample_time:.2f} samples/second")
    print("="*60)

if __name__ == "__main__":
    main()