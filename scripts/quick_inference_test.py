import torch
import time
import numpy as np
from CHiLD.modules.instantaneous import InstantaneousProcess
from CHiLD.tools.utils import load_yaml
import warnings
warnings.filterwarnings('ignore')

def test_inference_time(config_name, device='cuda', batch_size=64, num_runs=50):
    """Test inference time with synthetic data"""
    
    # Load config
    config_path = f'../CHiLD/configs/{config_name}.yaml'
    cfg = load_yaml(config_path)
    
    print(f"Testing dataset {config_name}")
    print(f"Config INPUT_DIM: {cfg['VAE']['INPUT_DIM']}")
    print(f"Config LATENT_DIM: {cfg['VAE']['LATENT_DIM']}")
    print(f"Config Z_DIM_LIST: {cfg['VAE']['Z_DIM_LIST']}")
    
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
    
    model.to(device)
    model.eval()
    
    # Create synthetic batch data matching config
    input_dim = cfg['VAE']['INPUT_DIM']
    latent_dim = cfg['VAE']['LATENT_DIM']
    n_mea = cfg['VAE']['N_MEA']  # Number of time steps needed
    
    # Create batch in the format expected by the model
    batch = {
        'xt': torch.randn(batch_size, n_mea, input_dim, device=device),
        'yt': torch.randn(batch_size, n_mea, latent_dim, device=device)
    }
    
    print(f"Input shape: {batch['xt'].shape}")
    print(f"Target shape: {batch['yt'].shape}")
    
    # Warmup runs
    print("Running warmup...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)
    
    # Timing runs
    print(f"Measuring inference time over {num_runs} runs...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(batch)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            batch_time = end_time - start_time
            times.append(batch_time)
            
            if (i + 1) % 10 == 0:
                print(f"Run {i+1}/{num_runs}: {batch_time:.4f}s")
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    per_sample_time = mean_time / batch_size
    
    print("\n" + "="*60)
    print(f"INFERENCE TIME RESULTS - Dataset {config_name}")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print(f"Device: {device}")
    print(f"Input dimensions: {input_dim}")
    print(f"Latent dimensions: {latent_dim}")
    print(f"Z_DIM_LIST: {cfg['VAE']['Z_DIM_LIST']}")
    print("-"*60)
    print(f"Mean time per batch:  {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"Min time per batch:   {min_time:.4f} seconds")
    print(f"Max time per batch:   {max_time:.4f} seconds")
    print(f"Mean time per sample: {per_sample_time*1000:.2f} milliseconds")
    print(f"Throughput:          {1/per_sample_time:.2f} samples/second")
    print("="*60)
    
    return per_sample_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['N', 'O'], help='Dataset to test')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_runs', type=int, default=50, help='Number of timing runs')
    
    args = parser.parse_args()
    
    test_inference_time(args.dataset, args.device, args.batch_size, args.num_runs)