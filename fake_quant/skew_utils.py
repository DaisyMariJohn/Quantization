import torch

def make_skew_hook(layer_name, stats_store):
    def hook(module, input, output):
        flat = output.detach().cpu().flatten()
        stats_store[layer_name] = {
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "min": flat.min().item(),
            "max": flat.max().item(),
            "skew": (((flat - flat.mean())**3).mean().item()) / (flat.std().item()**3 + 1e-6)
        }
    return hook

def analyze_skew(model, calib_loader, target_layer_names, device="cuda"):
    hooks = []
    stats_store = {}

    # ✅ Make sure model is on the right device
    model.to(device)

    for name, module in model.named_modules():
        if any(t in name for t in target_layer_names):
            hook_fn = make_skew_hook(name, stats_store)
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        inp, _ = calib_loader[0]
        inp = inp.to(device)  # ✅ move input to the same device
        model(inp)            # Run model forward pass

    for h in hooks:
        h.remove()

    return stats_store
