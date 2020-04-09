import standard_grid

if __name__=="__main__":
    grid = standard_grid.Grid("/work/sbenoit/repos/StrataNet/Experiments/MusicGeneration/run_hierarchical.py",
                              "/results/sbenoit/StrataNet/MusicGeneration/hierarchical/")
    # grid.register("batch_size", [1])
    grid.register("epochs", [1])
    grid.register("early_stop_threshold", [100])
    grid.register("model_lr", [1e-3])
    grid.register("latent_lr", [1e-3])
    grid.register("latent_dim", [1024])
    grid.register("latent_dims", ["1024,256,64,16"])
    grid.register("tier_lengths", ["32,1024"])
    grid.register("num_tiers", [3])
    grid.register("num_layers", [6])
    grid.register("dropout", [0.1])
    grid.register("block_size", [50])
    grid.register("data_dir", ["/results/sbenoit/datasets/lpd_processed/"])
    grid.generate_grid()
    grid.shuffle_grid()
    grid.generate_shell_instances(prefix="python3 ", postfix=">& stdoutput.txt")
    # total_at_a_time = 8
    grid.create_runner(num_runners=None, runners_prefix=["sbatch -p gpu_highmem -c 1 --gres=gpu:1"])
