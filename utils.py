import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

EVAL_BATCH_SIZE = 50


def normalize_image(image):
    """[0, 1] -> [-1, 1]"""
    return 2.0 * np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) - 1.0


def denormalize_image(image):
    """[-1, 1] -> [0, 1]"""
    return (np.clip(np.asarray(image, dtype=np.float32), -1.0, 1.0) + 1.0) / 2.0


def create_training_figure(input_img, target_img):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.12)
    images = [input_img, target_img, np.zeros_like(input_img)]
    for ax, image, title in zip(axes, images, ["Input", "Target", "Generated"]):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    return fig, axes[2].images[0]


def make_frame(fig, generated_im, generated, epoch):
    generated_im.set_data(generated)
    generated_im.axes.set_title(f"Generated (epoch {epoch})")
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())


def save_gif(frames, path, duration=500):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


@torch.no_grad()
def evaluate(predict, x, y, batch_size=EVAL_BATCH_SIZE):
    """RMSE over the whole set in [0, 1] image range, plus the first prediction."""
    squared_error, first = 0.0, None
    for i in range(0, len(x), batch_size):
        pred = denormalize_image(predict(x[i:i + batch_size]).cpu().numpy())
        target = denormalize_image(y[i:i + batch_size].cpu().numpy())
        squared_error += float(((pred - target) ** 2).sum())
        first = pred[0, 0] if first is None else first
    return float(np.sqrt(squared_error / y.numel())), first


@torch.no_grad()
def predict_all(predict, x, batch_size=EVAL_BATCH_SIZE):
    for i in range(0, len(x), batch_size):
        predict(x[i:i + batch_size])
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_memory_mb(fn):
    """Extra GPU memory (MB) needed while running fn, on top of what is already allocated. None on CPU."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2 ** 20


def measure_inference(predict, x):
    """Returns (ms per image, peak extra memory in MB) for predicting all of x."""
    start = time.perf_counter()
    predict_all(predict, x)
    ms_per_image = (time.perf_counter() - start) * 1000 / len(x)
    return ms_per_image, peak_memory_mb(lambda: predict_all(predict, x))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_rmse_curves(results, baseline_rmse, path="rmse_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, result in results.items():
        ax.plot(*zip(*result["rmse_history"]), marker="o", label=name)
    ax.axhline(baseline_rmse, linestyle="--", color="gray", label="low-res input")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test RMSE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def format_memory(mb):
    return "n/a" if mb is None else f"{mb:.0f} MB"


def print_summary(results, baseline_rmse):
    print(f"\nBaseline RMSE (low-res input): {baseline_rmse:.4f}")
    for name, r in results.items():
        print(
            f"{name:10s} rmse={r['rmse']:.4f} | params={r['params']:,} | "
            f"training: {r['train_time_s'] / 60:.1f} min, {format_memory(r['train_mem_mb'])} | "
            f"inference: {r['infer_ms_per_image']:.1f} ms/image, {format_memory(r['infer_mem_mb'])}"
        )