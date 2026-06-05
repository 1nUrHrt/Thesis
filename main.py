import torch

from Evaluate import run_experiment


if __name__ == "__main__":
    run_experiment("./configs")
    # file_name = "attn-res-gin-ddi"
    # best = torch.load(f"./checkpoints/{file_name}/best.pt", weights_only=False)
    # current = torch.load(f"./checkpoints/{file_name}/current.pt", weights_only=False)
    # best["classifier"] = best["decoder"]
    # current["classifier"] = current["decoder"]
    # del best["decoder"]
    # del current["decoder"]
    # torch.save(best, f"./checkpoints/{file_name}/best.pt")
    # torch.save(current, f"./checkpoints/{file_name}/current.pt")
