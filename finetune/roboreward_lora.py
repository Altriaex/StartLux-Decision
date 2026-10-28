"""Train StartLux AutoThink on the original RoboReward train/val directories.

Run with torchrun; use --max-steps for a bounded development run. The loss is
supervised option-letter cross entropy, not thought generation or RL.
Heavy imports are delayed until the experiment's runtime/cache paths are set.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def read_records(root, limit, seed, length_file=None):
    """Read metadata and optional precomputed lengths before selecting records."""
    lengths = json.loads(length_file.read_text()) if length_file is not None else None
    records = []
    with (root / "metadata.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            # Keep labels separate and never retain the label-verification explanation.
            record = {key: record[key] for key in ("file_name", "task", "reward")}
            if lengths is not None:
                record["token_length"] = lengths[index]
            records.append(record)
    if limit and limit < len(records):
        rng = random.Random(seed)
        groups = defaultdict(list)
        for record in records:
            groups[(record["file_name"].split("/")[0], record["reward"])].append(record)
        for group in groups.values():
            rng.shuffle(group)
        keys = list(groups)
        rng.shuffle(keys)
        records = []
        while len(records) < limit:
            for key in keys:
                if groups[key]:
                    records.append(groups[key].pop())
                    if len(records) == limit:
                        break
    return records


class VideoBatch:
    """Render the same image-first decision as the HELM endpoint client."""

    def __init__(self, root, processor, spec):
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
            Qwen3VLVideoProcessor,
        )

        self.root, self.processor, self.spec = root, processor, spec
        self.video_processor = Qwen3VLVideoProcessor(fps=2, min_frames=4, max_frames=8)

    def __call__(self, records):
        import torch
        from PIL import Image
        from transformers.video_utils import read_video_torchcodec

        from startlux_decide.images import place_images
        from startlux_decide.prompt import from_systemone, messages, plain_prompt

        texts, images = [], []
        for record in records:
            # Keep deployed sampling and original RGB pixels without media serialization.
            frames, _ = read_video_torchcodec(
                str(self.root / record["file_name"]), self.video_processor.sample_frames
            )
            state = place_images("Task: " + record["task"], len(frames))
            row = from_systemone(state, self.spec)
            texts.append(plain_prompt(self.processor.tokenizer, messages(row)))
            # Frames are [F, C, H, W]; each PIL image receives [H, W, C].
            images.extend(
                Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in frames
            )
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            size={"longest_edge": 1048576, "shortest_edge": 65536},
            return_tensors="pt",
        )
        inputs = {
            key: batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "mm_token_type_ids",
            )
        }
        targets = torch.tensor([record["reward"] - 1 for record in records])
        return inputs, targets, [record["file_name"] for record in records]


def build_model(path, device, args):
    """Keep the multimodal forward, with adapters only inside the text decoder."""
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    processor = transformers.AutoProcessor.from_pretrained(path)
    processor.tokenizer.padding_side = "right"
    with (path / "decision_config.json").open(encoding="utf-8") as stream:
        letters = json.load(stream)["letter_token_ids"][:5]
    config = transformers.AutoConfig.from_pretrained(path)
    full = getattr(transformers, config.architectures[0]).from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
    )
    full.requires_grad_(False)
    if args.gradient_checkpointing:
        full.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    full.model.language_model = get_peft_model(
        full.model.language_model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules="all-linear",
        ),
    )

    class DecisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = full.model
            head = full.get_output_embeddings().weight
            self.register_buffer("letter_rows", head[letters].detach().float().clone())

        def forward(self, inputs):
            # [B, L, H] -> [B, H] -> [B, 5], without materializing vocabulary logits.
            hidden = self.backbone(
                **inputs, use_cache=False, return_dict=True
            ).last_hidden_state
            last = inputs["attention_mask"].sum(-1) - 1
            h = hidden[
                torch.arange(hidden.shape[0], device=hidden.device), last
            ].float()
            return h @ self.letter_rows.T

    return full, DecisionModel(), processor


def evaluate(
    model, records, collate, batch_size, device, rank, world, output, num_workers
):
    """Shard without padding, then reduce sample sums (not per-rank averages)."""
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    predictions = []
    loader = DataLoader(
        records[rank::world],
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
        multiprocessing_context="spawn" if num_workers else None,
    )
    with torch.no_grad():
        for inputs, target, names in loader:
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in inputs.items()
            }
            target = target.to(device, non_blocking=True)
            logits = model(inputs)
            predicted = logits.argmax(-1)
            error = (predicted - target).float()
            totals += torch.stack(
                (
                    F.cross_entropy(logits, target, reduction="sum"),
                    (predicted == target).sum(),
                    error.abs().sum(),
                    error.square().sum(),
                    torch.tensor(len(target), device=device),
                )
            ).double()
            probabilities = logits.softmax(-1).cpu().tolist()
            for name, label, guess, probability in zip(
                names, target.tolist(), predicted.tolist(), probabilities
            ):
                predictions.append(
                    {
                        "sample_id": name,
                        "reward": label + 1,
                        "prediction": guess + 1,
                        "probabilities": probability,
                    }
                )
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(predictions, gathered, dst=0)
    dist.all_reduce(totals)
    if rank == 0:
        with output.open("w", encoding="utf-8") as stream:
            for shard in gathered:
                for prediction in shard:
                    stream.write(json.dumps(prediction) + "\n")
    n = totals[-1].item()
    return {
        "val/" + key: value / n
        for key, value in zip(("loss", "accuracy", "mae", "mse"), totals[:-1].tolist())
    }


if __name__ == "__main__":
    # exp_name is caller-supplied: include dataset, model and experiment identifiers.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-name", required=True, help="e.g. rr-startlux4b-lr5e5 (rr=RoboReward)"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=4, help="samples per GPU micro-batch"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--group-by-length", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument(
        "--max-steps", type=int, help="bound optimizer updates for development runs"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="0 reads all original train records",
    )
    parser.add_argument(
        "--val-samples", type=int, default=500, help="0 reads all original val records"
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--warmup", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-every", type=int, default=0, help="0 saves every half epoch"
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--swanlab-project", default="startlux-roboreward")
    parser.add_argument("--swanlab-workspace")
    parser.add_argument(
        "--swanlab-mode", choices=("online", "offline"), default="online"
    )
    args = parser.parse_args()
    if (
        not args.exp_name
        or Path(args.exp_name).name != args.exp_name
        or args.exp_name in (".", "..")
    ):
        parser.error("exp-name must be a single directory name")
    args.model = args.model.resolve()
    args.data_root = args.data_root.resolve()
    if args.resume:
        args.resume = args.resume.resolve()

    # Configure all writable runtime paths before importing CUDA or tracking libraries.
    repo = Path(__file__).resolve().parents[1]
    output = repo.parent / "experiments" / args.exp_name
    for child in ("logs", "logs/swanlab", "checkpoints", "validation", "tmp"):
        (output / child).mkdir(parents=True, exist_ok=True)
    for key, child in {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "tmp/xdg",
        "HF_HOME": "tmp/huggingface",
        "TORCH_HOME": "tmp/torch",
        "PYTORCH_KERNEL_CACHE_PATH": "tmp/torch-kernels",
        "TORCHINDUCTOR_CACHE_DIR": "tmp/inductor",
        "TRITON_CACHE_DIR": "tmp/triton",
        "CUDA_CACHE_PATH": "tmp/cuda",
        "SWANLAB_ROOT": "tmp/swanlab-root",
    }.items():
        os.environ[key] = str(output / child)
        (output / child).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    os.chdir(output)

    import importlib.metadata

    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    # Initialize rank-local CUDA and read the original dataset on each rank.
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s rank={rank} %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "logs" / f"rank-{rank}.log"),
        ],
    )
    logger = logging.getLogger("roboreward")
    if rank == 0:
        train_handler = logging.FileHandler(output / "logs/train.log")
        train_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(train_handler)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if (output / "config.json").exists() and args.resume is None:
        raise ValueError(
            "Experiment already exists; choose another exp-name or explicitly resume a checkpoint"
        )
    train = read_records(
        args.data_root / "train",
        args.train_samples,
        args.seed,
        args.data_root / "train/startlux4b-token-lengths.json"
        if args.group_by_length
        else None,
    )
    val = read_records(args.data_root / "val", args.val_samples, args.seed)

    # The rubric is the official HELM RoboReward prompt mapped by StartLuxEndpointClient.
    spec = {
        "type": "choice",
        "instructions": (
            "Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video.\n"
            "Rubric for end-of-episode progress (judge only the final state without time limits):"
        ),
        "criteria": {
            "1": "1 - No Success: Final state shows no goal-relevant change for the command.",
            "2": "2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.",
            "3": "3 - Partial Completion: The final state shows good progress toward the goal but violates "
            "more than one requirement or a major requirement.",
            "4": "4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.",
            "5": "5 - Perfect Completion: Final state satisfies all requirements.",
        },
    }
    loading = time.perf_counter()
    full, module, processor = build_model(args.model, device, args)
    module.to(device)
    torch.cuda.synchronize(device)
    logger.info("model_load_seconds=%.3f", time.perf_counter() - loading)
    train_collate = VideoBatch(args.data_root / "train", processor, spec)
    val_collate = VideoBatch(args.data_root / "val", processor, spec)
    if args.group_by_length:
        train_lengths = [record["token_length"] for record in train]

    sampler = DistributedSampler(
        train, num_replicas=world, rank=rank, seed=args.seed, drop_last=False
    )
    indices = list(sampler)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        sampler=indices,
        collate_fn=train_collate,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
        multiprocessing_context="spawn" if args.num_workers else None,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = DistributedDataParallel(
        module, device_ids=[local_rank], broadcast_buffers=False
    )
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
    batches_per_epoch = len(loader)
    steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warm = max(1, int(args.warmup * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warm)
            * 0.5
            * (1 + math.cos(math.pi * step / total_steps))
        ),
    )
    save_every = args.save_every or max(1, math.ceil(steps_per_epoch / 2))
    step, start_epoch, start_batch, swanlab_id = 0, 0, 0, None

    # Resume actual training state; tracking state alone never restores a model.
    if args.resume:
        state = torch.load(
            args.resume / "training.pt", map_location="cpu", weights_only=False
        )
        if len(state["rng"]) != world:
            raise ValueError(
                "Resume requires the same GPU count to preserve the sampler position"
            )
        for key in (
            "exp_name",
            "model",
            "data_root",
            "batch_size",
            "gradient_accumulation_steps",
            "train_samples",
            "seed",
            "epochs",
            "max_steps",
            "lr",
            "warmup",
            "rank",
            "alpha",
            "dropout",
            "group_by_length",
        ):
            if config[key] != state["config"][key]:
                raise ValueError(f"Resume changes data/optimization position: {key}")
        set_peft_model_state_dict(module.backbone.language_model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, start_batch = (
            state["step"],
            state["epoch"],
            state["next_batch"],
        )
        swanlab_id = state["swanlab_id"]
        rng = state["rng"][rank]

    # Persist the exact sources, selections and environment inside this experiment.
    run = None
    if rank == 0:
        import swanlab

        environment = {
            "commit": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_diff": subprocess.check_output(
                ["git", "-C", str(repo), "diff"], text=True
            ),
            "packages": {
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "transformers",
                    "peft",
                    "swanlab",
                    "torchcodec",
                    "flash-linear-attention",
                    "fla-core",
                    "causal-conv1d",
                    "triton",
                )
            },
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "train_records": len(train),
            "val_records": len(val),
            "sampler_padding_per_epoch": len(sampler) * world - len(train),
            "train_labels": dict(Counter(r["reward"] for r in train)),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "adapter_modules": [
                name for name, p in module.named_parameters() if p.requires_grad
            ],
        }
        shutil.copyfile(__file__, output / "logs" / "roboreward_lora.py")
        for name, records in (("train", train), ("val", val)):
            (output / "logs" / f"{name}-samples.json").write_text(
                json.dumps([r["file_name"] for r in records]), encoding="utf-8"
            )
        run = swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            name=args.exp_name,
            log_dir=str(output / "logs/swanlab"),
            mode=args.swanlab_mode,
            id=swanlab_id if args.resume else None,
            resume="must"
            if args.resume and args.swanlab_mode == "online"
            else "allow"
            if args.resume
            else "never",
            config=config,
            settings=swanlab.Settings(interactive=False),
        )
        swanlab_id = run.id
        environment["swanlab_id"] = swanlab_id
        (output / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        (output / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        logger.info("configuration=%s", json.dumps(config))
        logger.info(
            "gpu=%s world_size=%s train_records=%s val_records=%s trainable_parameters=%s",
            environment["gpu"],
            world,
            len(train),
            len(val),
            environment["trainable_parameters"],
        )
    identity = [swanlab_id]
    dist.broadcast_object_list(identity, src=0)
    swanlab_id = identity[0]
    # SDK initialization must not perturb the random state restored from a checkpoint.
    if args.resume:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
    else:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    optimizer.zero_grad(set_to_none=True)

    # Update only the adapters; accumulation windows use their actual sample count.
    try:
        for epoch in range(start_epoch, math.ceil(args.epochs)):
            # Rebuild this rank's epoch order before applying the resume batch offset.
            sampler.set_epoch(epoch)
            indices[:] = list(sampler)
            if args.group_by_length:
                batches = []
                window = 50 * args.batch_size
                for begin in range(0, len(indices), window):
                    ordered = sorted(
                        indices[begin : begin + window], key=train_lengths.__getitem__
                    )
                    batches.extend(
                        ordered[i : i + args.batch_size]
                        for i in range(0, len(ordered), args.batch_size)
                    )
                random.Random(args.seed + epoch).shuffle(batches)
                indices[:] = [index for batch in batches for index in batch]
            first_batch = start_batch if epoch == start_epoch else 0
            indices[:] = indices[first_batch * args.batch_size :]
            opening = time.perf_counter()
            iterator = iter(loader)
            logger.info(
                "epoch=%s first_batch=%s iterator_start_seconds=%.3f",
                epoch,
                first_batch,
                time.perf_counter() - opening,
            )
            batch_index = first_batch
            while batch_index < batches_per_epoch and step < total_steps:
                module.train()
                module.backbone.visual.eval()
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                n_micro = min(
                    args.gradient_accumulation_steps, batches_per_epoch - batch_index
                )
                n_samples = min(
                    n_micro * args.batch_size,
                    len(sampler) - batch_index * args.batch_size,
                )
                loss_sum = torch.zeros((), device=device)
                lengths, padded_tokens, data_wait = [], 0, 0.0
                events = [
                    [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    for _ in range(n_micro)
                ]
                for micro in range(n_micro):
                    waiting = time.perf_counter()
                    inputs, target, _ = next(iterator)
                    data_wait += time.perf_counter() - waiting
                    lengths.extend(inputs["attention_mask"].sum(-1).tolist())
                    padded_tokens += inputs["input_ids"].numel()
                    events[micro][0].record()
                    inputs = {
                        key: value.to(device, non_blocking=True)
                        for key, value in inputs.items()
                    }
                    target = target.to(device, non_blocking=True)
                    events[micro][1].record()
                    sync = (
                        contextlib.nullcontext()
                        if micro == n_micro - 1
                        else model.no_sync()
                    )
                    with sync:
                        logits = model(inputs)
                        loss = F.cross_entropy(logits, target, reduction="sum")
                        events[micro][2].record()
                        (loss / n_samples).backward()
                        events[micro][3].record()
                    loss_sum += loss.detach()
                    batch_index += 1
                optimizer_events = [
                    torch.cuda.Event(enable_timing=True) for _ in range(2)
                ]
                optimizer_events[0].record()
                norm = torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_events[1].record()
                step += 1
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                if step % args.log_every == 0 or step == total_steps:
                    sums = torch.tensor(
                        [loss_sum.item(), n_samples, sum(lengths), padded_tokens],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(sums)
                    elapsed = torch.tensor(seconds, device=device)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    stages = torch.tensor(
                        [
                            data_wait,
                            *(
                                sum(e[i].elapsed_time(e[i + 1]) for e in events) / 1000
                                for i in range(3)
                            ),
                            optimizer_events[0].elapsed_time(optimizer_events[1])
                            / 1000,
                        ],
                        device=device,
                    )
                    dist.all_reduce(stages, op=dist.ReduceOp.MAX)
                    memory = torch.tensor(
                        [
                            torch.cuda.max_memory_allocated(device) / 2**30,
                            torch.cuda.max_memory_reserved(device) / 2**30,
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                        ],
                        device=device,
                    )
                    peaks = [torch.zeros_like(memory) for _ in range(world)]
                    dist.all_gather(peaks, memory)
                    metrics = {
                        "train/loss": (sums[0] / sums[1]).item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + batch_index / batches_per_epoch,
                        "train/effective_batch": sums[1].item(),
                        "train/gradient_norm": norm.item(),
                        "train/max_local_tokens": max(lengths),
                        "perf/step_seconds": elapsed.item(),
                        "perf/samples_per_second": sums[1].item() / elapsed.item(),
                        "perf/valid_tokens_per_second": sums[2].item() / elapsed.item(),
                        "perf/padded_to_valid_ratio": (sums[3] / sums[2]).item(),
                        **{
                            f"perf/{name}_seconds": value
                            for name, value in zip(
                                (
                                    "data_wait",
                                    "h2d",
                                    "forward",
                                    "backward",
                                    "optimizer",
                                ),
                                stages.tolist(),
                            )
                        },
                        **{
                            f"memory/rank{i}_{name}": value
                            for i, peak in enumerate(peaks)
                            for name, value in zip(
                                ("peak_gb", "reserved_gb", "parent_peak_rss_gb"),
                                peak.tolist(),
                            )
                        },
                    }
                    if rank == 0:
                        logger.info("step=%s metrics=%s", step, json.dumps(metrics))
                        with (output / "logs/metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps({"step": step, **metrics}) + "\n")
                        run.log(metrics, step=step)

                # Validate and checkpoint at the same completed optimizer boundary on all ranks.
                if step % save_every == 0 or step == total_steps:
                    evaluating = time.perf_counter()
                    metrics = evaluate(
                        module,
                        val,
                        val_collate,
                        args.batch_size,
                        device,
                        rank,
                        world,
                        output / "validation" / f"step-{step:06d}.jsonl",
                        args.num_workers,
                    )
                    metrics["perf/validation_seconds"] = (
                        time.perf_counter() - evaluating
                    )
                    rng = {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(device),
                    }
                    rng_states = [None] * world if rank == 0 else None
                    dist.gather_object(rng, rng_states, dst=0)
                    if rank == 0:
                        logger.info("step=%s validation=%s", step, json.dumps(metrics))
                        with (output / "logs/metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps({"step": step, **metrics}) + "\n")
                        run.log(metrics, step=step)
                        saving = time.perf_counter()
                        checkpoint = output / "checkpoints" / f"step-{step:06d}"
                        checkpoint.mkdir()
                        module.backbone.language_model.save_pretrained(
                            checkpoint / "adapter"
                        )
                        torch.save(
                            {
                                "adapter": get_peft_model_state_dict(
                                    module.backbone.language_model
                                ),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "step": step,
                                "epoch": epoch,
                                "next_batch": batch_index,
                                "rng": rng_states,
                                "swanlab_id": swanlab_id,
                                "config": config,
                            },
                            checkpoint / "training.pt",
                        )
                        logger.info(
                            "step=%s checkpoint_seconds=%.3f",
                            step,
                            time.perf_counter() - saving,
                        )
                    dist.barrier()
            if step >= total_steps:
                break

        # Release persistent workers before exporting the full model on the host.
        del iterator, loader

        # Rank zero exports a usable model; all other ranks wait until export finishes.
        if rank == 0:
            exporting = time.perf_counter()
            final = output / "final"
            if final.exists():
                raise FileExistsError(
                    "Final export already exists; use a new experiment for a new export"
                )
            module.backbone.language_model.save_pretrained(final / "adapter")
            full.model.language_model = (
                module.backbone.language_model.merge_and_unload()
            )
            full.save_pretrained(final / "merged", safe_serialization=True)
            processor.save_pretrained(final / "merged")
            shutil.copyfile(
                args.model / "decision_config.json",
                final / "merged/decision_config.json",
            )
            logger.info(
                "finished step=%s final=%s export_seconds=%.3f",
                step,
                final,
                time.perf_counter() - exporting,
            )
        dist.barrier()
    finally:
        if run is not None:
            run.finish()
        dist.destroy_process_group()
