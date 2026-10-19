import json
import time

import numpy as np
import torch

from data_generation import generate_data
from diffusion import build_diffusion, train_step as diffusion_step
from gan import Generator, PatchDiscriminator, train_step as gan_step
from utils import (
    count_params, create_training_figure, denormalize_image, evaluate, make_frame, measure_inference,
    normalize_image, peak_memory_mb, plot_rmse_curves, print_summary, save_gif,
)

NUM_TRAIN = 1000
NUM_TEST = 100
WIDTH = 64
HEIGHT = 64
BATCH_SIZE = 8
NUM_EPOCHS = 100
EVAL_EVERY = 5
SAMPLE_STEPS = 100
SAVE_PATH = "perlin_data.npz"
GEN_SAMPLES = True
SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    num_samples = NUM_TRAIN + NUM_TEST
    if GEN_SAMPLES:
        y, x = generate_data(num_samples, WIDTH, HEIGHT, SAVE_PATH, SEED)
    else:
        data = np.load(SAVE_PATH)
        y, x = data["original"], data["downscaled"]
    assert len(x) >= num_samples, "not enough samples in the file, set GEN_SAMPLES = True"

    x = torch.from_numpy(normalize_image(x[:num_samples])).unsqueeze(1).to(DEVICE)
    y = torch.from_numpy(normalize_image(y[:num_samples])).unsqueeze(1).to(DEVICE)
    return x[:NUM_TRAIN], y[:NUM_TRAIN], x[NUM_TRAIN:], y[NUM_TRAIN:]


def setup_gan():
    generator = Generator().to(DEVICE)
    discriminator = PatchDiscriminator().to(DEVICE)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    step = lambda x, y: gan_step(generator, discriminator, optimizer_G, optimizer_D, x, y)
    predict = lambda x: generator(x)
    return step, predict, generator


def setup_diffusion():
    diffusion = build_diffusion().to(DEVICE)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
    step = lambda x, y: diffusion_step(diffusion, optimizer, x, y)
    predict = lambda x: diffusion.sample(x, steps=SAMPLE_STEPS)
    return step, predict, diffusion


def train(name, step, predict, data):
    train_x, train_y, test_x, test_y = data
    input_img = denormalize_image(test_x[0, 0].cpu().numpy())
    target_img = denormalize_image(test_y[0, 0].cpu().numpy())
    fig, generated_im = create_training_figure(input_img, target_img)
    frames, rmse_history, train_time = [], [], 0.0

    def evaluate_epoch(epoch):
        rmse, generated = evaluate(predict, test_x, test_y)
        frames.append(make_frame(fig, generated_im, generated, epoch))
        return rmse

    evaluate_epoch(0)  # untrained model, first frame of the GIF
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.perf_counter()
        order = torch.randperm(len(train_x), device=DEVICE)
        losses = [step(train_x[idx], train_y[idx]) for idx in order.split(BATCH_SIZE)]
        train_time += time.perf_counter() - start

        log = " | ".join(f"{key}={np.mean([loss[key] for loss in losses]):.4f}" for key in losses[0])
        if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
            rmse = evaluate_epoch(epoch)
            rmse_history.append((epoch, rmse))
            log += f" | test_rmse={rmse:.4f}"
        print(f"[{name}] epoch {epoch:03d}/{NUM_EPOCHS} | {log}", flush=True)

    save_gif(frames, f"{name}_training.gif")
    frames[-1].save(f"{name}_last_generation.png")

    infer_ms, infer_mem = measure_inference(predict, test_x)
    return {
        "rmse": rmse_history[-1][1],
        "rmse_history": rmse_history,
        "train_time_s": train_time,
        "train_mem_mb": peak_memory_mb(lambda: step(train_x[:BATCH_SIZE], train_y[:BATCH_SIZE])),
        "infer_ms_per_image": infer_ms,
        "infer_mem_mb": infer_mem,
    }


def main():
    data = load_data()
    baseline_rmse, _ = evaluate(lambda x: x, data[2], data[3])
    print(f"device: {DEVICE} | baseline RMSE (low-res input): {baseline_rmse:.4f}")

    results = {}
    for name, setup in {"gan": setup_gan, "diffusion": setup_diffusion}.items():
        torch.manual_seed(SEED)
        step, predict, model = setup()
        results[name] = train(name, step, predict, data)
        results[name]["params"] = count_params(model)

    print_summary(results, baseline_rmse)
    plot_rmse_curves(results, baseline_rmse)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()