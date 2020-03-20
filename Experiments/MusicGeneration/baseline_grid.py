import standard_grid

if __name__=="__main__":
    grid = standard_grid.Grid("/work/sbenoit/repos/StrataNet/Experiments/MusicGeneration/run_baseline.py", "/results/sbenoit/StrataNet/MusicGeneration/baseline/")
    # grid.register("batch_size", [1])
    grid.register("epochs", [1])
    grid.register("early_stop_threshold", [100])
    grid.register("model_lr", [1e-3])
    grid.register("latent_lr", [1e-3])
    grid.register("latent_dim", [128])
    grid.register("n_layers", [6])
    grid.register("d_ff", [1024])
    grid.register("n_heads", [8])
    grid.register("dropout", [0.1])
    grid.register("data_dir", ["/results/sbenoit/datasets/lpd_processed/"])
    grid.generate_grid()
    grid.shuffle_grid()
    grid.generate_shell_instances(prefix="python3 ", postfix=">& stdoutput.txt")
    # total_at_a_time = 8
    grid.create_runner(num_runners=None, runners_prefix=["sbatch -p gpu_high -c 1 --gres=gpu:1"])
