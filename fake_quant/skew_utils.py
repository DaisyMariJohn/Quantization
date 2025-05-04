import torch

# Create a forward hook function that records output statistics (mean, std, min, max, skewness)
def make_skew_hook(layer_name, stats_store):
    def hook(module, input, output):
        # Flatten the output tensor to 1D for statistical computation
        flat = output.detach().cpu().flatten()
        
        # Store various stats about the output activations for this layer
        stats_store[layer_name] = {
            "mean": flat.mean().item(),                                 # Mean of output activations
            "std": flat.std().item(),                                   # Standard deviation
            "min": flat.min().item(),                                   # Minimum value
            "max": flat.max().item(),                                   # Maximum value
            "skew": (((flat - flat.mean())**3).mean().item()) /         # Skewness: how asymmetric the distribution is
                    (flat.std().item()**3 + 1e-6)                        # Add small epsilon to avoid division by zero
        }
    return hook

# Analyze the skew of activation distributions for target layers using a calibration batch
def analyze_skew(model, calib_loader, target_layer_names, device="cuda"):
    hooks = []          # To store hook handles for later removal
    stats_store = {}    # Dictionary to store layer-wise stats

    # Register forward hooks on layers whose names match any of the target layer names
    for name, module in model.named_modules():
        if any(t in name for t in target_layer_names):
            hook_fn = make_skew_hook(name, stats_store)
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()  # Set the model to evaluation mode (no dropout, etc.)

    # Perform a forward pass using the first batch from the calibration data
    with torch.no_grad():  # Disable gradient tracking
        inp, _ = calib_loader[0]         # Get first batch (input and label; ignore label)
        model(inp.to(device))            # Run the model to trigger hooks and collect stats

    # Remove all registered hooks to prevent memory leaks and side effects
    for h in hooks:
        h.remove()

    return stats_store  # Return the collected stats for each target layer
