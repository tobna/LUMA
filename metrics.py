from contextlib import nullcontext
from metrics_ops import flops, macs

from time import time

import psutil
import torch.cuda
import torch.distributed as dist


from tqdm.auto import tqdm

# ── windowed-inference val-set histograms ────────────────────────────────────
# {num_crops_per_image: number_of_images} for each val set. num_crops is
# ceil(longer_side / shorter_side) under the windowing in
# training/lightning_module.py::window_imgs_semantic (shorter side resized
# to min(img_size), then tiled at min(img_size) stride). Independent of
# img_size since only the original aspect ratio matters.
_VAL_CROP_HIST = {
    "ade20k_val": {1: 347, 2: 1648, 3: 5},  # 2000 images total
    "cityscapes_val": {2: 500},  # 500 images, all 1024×2048
}


def _weighted_mean(per_bs, hist):
    """Σ n_c · per_bs[c] / Σ n_c — mean of a per-batch-size value over the histogram."""
    n_total = sum(hist.values())
    if any(per_bs[c] is None for c in hist.keys()):
        return None
    return sum(n * per_bs[c] for c, n in hist.items()) / n_total


class _disabled_fused_attention:
    """Context manager: temporarily flip every `fused_attn` flag to False.

    fvcore's FlopCountAnalysis (and torchprofile) cannot see inside timm's
    fused `scaled_dot_product_attention` call (it dispatches to FlashAttention
    / mem-efficient SDPA at the C++ level), so the QKᵀ matmul and softmax·V
    matmul are silently dropped from the FLOPs count. Disabling the fused
    path forces the explicit `softmax(QK^T/√d)·V` implementation, which the
    tracers count correctly. The original state is restored on exit so
    throughput timing (next) still uses Flash/SDPA.
    """

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        for m in self.model.modules():
            if hasattr(m, "fused_attn"):
                self.saved.append((m, m.fused_attn))
                m.fused_attn = False
        return self

    def __exit__(self, *exc):
        for m, prev in self.saved:
            m.fused_attn = prev


def calculate_metrics(
    args,
    model,
    rank=0,
    input=None,
    device=None,
    did_training=False,
    world_size=1,
    all_metrics=True,
    n_ims=1,
    optim=None,
    scaler=None,
    train_loader=None,
    key_start="eval/",
    max_batch_size=None,
):
    """Calculate all metrics.

    Args:
        args: training arguments; in particular set args.eval_amp
        model (torch.nn.Module): model to analyze
        rank (int, optional): rank of this process (Default value = 0)
        input (torch.Tensor, optional): input batch (Default value = None)
        device (torch.device, optional): device to calculate throughput on (Default value = None)
        did_training (bool, optional): call after training to measure peak memory usage (Default value = False)
        world_size (int, optional): number of processes/GPUs for peak memory usage (Default value = 1)
        all_metrics (bool, optional): flag to calculate all metrics. If false, only number of parameters and memory usage is calculated. (Default value = True)
        n_ims (int, optional): number of images to consider for macs and flops (Default value = 1)

    Returns:
        dict: dictionary of metrics

    """
    assert 0 <= rank < world_size, f"Incompatible rank and world size; not 0 <= rank={rank} < world_size={world_size}"
    if rank != 0:
        max_mem_allocated(device, world_size)
        return {}

    assert (
        input is not None or train_loader is not None
    ), "Set either input tensor or train_loader to have some data for metrics calculation"
    if input is None:
        input = next(iter(train_loader))[0].to(device)

    if input.size(0) == 1 and args.batch_size > 1:
        input = input[0]

    print(f"Calculating metrics on input of shape {input.shape}.")

    metrics = {key_start + "number of parameters": number_of_params(model)}
    if input is None:
        return metrics

    if did_training:
        if world_size == 1:
            metrics[key_start + "peak_memory"] = max_mem_allocated(device, world_size)
        else:
            peak_mem_total, peak_mem_single = max_mem_allocated(device, world_size)
            metrics[key_start + "peak_memory_total"] = peak_mem_total
            metrics[key_start + "peak_memory_single"] = peak_mem_single

    if not all_metrics:
        return metrics

    model.eval()
    if args.cuda:
        for bs, inference_mem in inference_memory(args, model, input, device).items():
            metrics[key_start + f"inference_memory_@{bs}"] = inference_mem

    # FLOPs / MACs at each per-image batch size used by the windowed val loop.
    # For a c-crop image, the val loop runs exactly one forward at bs=c, so
    # flops_at_c is the total ops for that image. The mean per image across
    # the val set is then Σ n_c · flops_at_c / N_total (see _weighted_mean).
    # Compute on the un-compiled module (torch.compile wraps under _orig_mod)
    # with fused SDPA disabled so fvcore/torchprofile actually see the
    # attention matmuls (see _disabled_fused_attention docstring).
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    with _disabled_fused_attention(raw_model):
        macs_at = {}
        flops_at = {}
        flops_plain_at = {}
        for c in (1, 2, 3, 4):
            if c > args.batch_size:
                flops_at[c] = macs_at[c] = flops_plain_at[c] = None
                continue
            try:
                macs_at[c] = macs(args, raw_model, input, n_ims=c)
            except (RuntimeError, torch.OutOfMemoryError) as e:
                print(f"Failed to calculate macs @ bs={c}: {e}")
                print("Setting macs to None")
                macs_at[c] = None
            try:
                flops_at[c] = flops(args, raw_model, input, n_ims=c)
            except RuntimeError as e:
                print(f"Failed to calculate flops @ bs={c}: {e}")
                print(f"Setting flops @ bs={c} equal to macs!")
                flops_at[c] = macs_at[c]
            try:
                flops_plain_at[c] = flops(args, raw_model, input, n_ims=c, plain=True)
            except RuntimeError as e:
                print(f"Failed to calculate plain flops @ bs={c}: {e}")
                print(f"Setting plain flops @ bs={c} equal to macs!")
                flops_plain_at[c] = macs_at[c]
    metrics[key_start + "macs"] = {
        "at_bs_1": macs_at[1],
        "at_bs_2": macs_at[2],
        "at_bs_3": macs_at[3],
        "at_bs_4": macs_at[4],
        **{name: _weighted_mean(macs_at, hist) for name, hist in _VAL_CROP_HIST.items()},
    }
    metrics[key_start + "flops"] = {
        "at_bs_1": flops_at[1],
        "at_bs_2": flops_at[2],
        "at_bs_3": flops_at[3],
        "at_bs_4": flops_at[4],
        **{name: _weighted_mean(flops_at, hist) for name, hist in _VAL_CROP_HIST.items()},
    }
    # Plain fvcore (default handlers only) — matches what most papers report.
    metrics[key_start + "flops_plain"] = {
        "at_bs_1": flops_plain_at[1],
        "at_bs_2": flops_plain_at[2],
        "at_bs_3": flops_plain_at[3],
        "at_bs_4": flops_plain_at[4],
        **{name: _weighted_mean(flops_plain_at, hist) for name, hist in _VAL_CROP_HIST.items()},
    }

    if device is None:
        return metrics

    if optim is None or scaler is None or train_loader is None:
        print(
            f"Skipping training time calculation, since one of these is None: optimizer={optim}, scaler={scaler},"
            f" train_loader={train_loader}"
        )
    else:
        print(f"Calculating training time for 200 steps at batch size {args.batch_size}")
        max_mem_allocated(device, world_size)
        train_time = training_time(
            args, model=model, optim=optim, scaler=scaler, data_loader=train_loader, device=device, max_iters=200
        )
        metrics[key_start + "training_time"] = {"batch_size": args.batch_size, "step_time_ms": train_time}
        if world_size == 1:
            metrics[key_start + "peak_memory"] = max_mem_allocated(device, world_size)
        else:
            peak_mem_total, peak_mem_single = max_mem_allocated(device, world_size)
            metrics[key_start + "peak_memory_total"] = peak_mem_total
            metrics[key_start + "peak_memory_single"] = peak_mem_single

    tp_bs, tp_at_1 = throughput(args, model, input, fixed_bs=1)
    assert tp_bs == 1
    tp_at_2 = tp_at_3 = tp_at_4 = None
    if args.batch_size >= 2:
        tp_bs, tp_at_2 = throughput(args, model, input, fixed_bs=2)
        assert tp_bs == 2
    if args.batch_size >= 3:
        tp_bs, tp_at_3 = throughput(args, model, input, fixed_bs=3)
        assert tp_bs == 3
    if args.batch_size >= 4:
        tp_bs, tp_at_4 = throughput(args, model, input, fixed_bs=4)
        assert tp_bs == 4
    tp_bs, tp_val = throughput(args, model, input, max_batch_size=max_batch_size)

    # ── paper-comparable FPS (windowed inference) ────────────────────────────
    # The val loop tiles each image into num_crops = ceil(long/short) crops
    # (see _VAL_CROP_HIST). Crops from one image are stacked into a single
    # forward, so each c-crop image takes T(c) = c / tp_at_c seconds.
    # Mean time-per-image = Σ n_c · T(c) / N_total ; FPS = 1 / mean.
    t_per_img_at = {
        1: (1 / tp_at_1) if tp_at_1 is not None else None,
        2: (2 / tp_at_2) if tp_at_2 is not None else None,
        3: (3 / tp_at_3) if tp_at_3 is not None else None,
        4: (4 / tp_at_4) if tp_at_4 is not None else None,
    }
    metrics[key_start + "throughput"] = {
        "batch_size": tp_bs,
        "value": tp_val,
        "at_bs_1": tp_at_1,
        "at_bs_2": tp_at_2,
        "at_bs_3": tp_at_3,
        "at_bs_4": tp_at_4,
        **{
            name: (1 / _weighted_mean(t_per_img_at, hist)) if _weighted_mean(t_per_img_at, hist) is not None else None
            for name, hist in _VAL_CROP_HIST.items()
        },
    }

    return metrics


def max_mem_allocated(device, world_size=1, reset_max=False):
    """Return the max memory allocated during training.

    Use **this before** calling *throughput*, as that resets the statistics.

    Args:
        device (torch.Device): the device to look at in this process
        world_size (int, optional): the number of GPUs (processes) used in total -> stats are gathered from all GPUs (Default value = 1)
        reset_max (bool, optional): if true, resets the max memory allocated to zero. Subsequent calls will return the max memory allocated after this call. (Default value = False)

    Returns:
        int: the max memory allocated during training

    """
    max_mem_gpu = torch.cuda.max_memory_allocated(device)
    if reset_max:
        torch.cuda.reset_peak_memory_stats(device)

    if world_size == 1:
        return max_mem_gpu

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, max_mem_gpu)
    return sum(gathered), max(gathered)


def number_of_params(model):
    """Get the number of parameters from the model.

    Args:
        model (torch.nn.Module): the model

    Returns:
        int: number of parameters

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def inference_memory(args, model, input, device, batch_sizes=(1, 2, 4, 16, 32, 64, 128)):
    """Return the memory needed for inference at different batch sizes.

    Args:
        args: training arguments; in particular set args.eval_amp
        model (torch.nn.Module): the model to evaluate
        input (torch.Tensor): batch of input data; no batch size bigger than the size of this batch are tested
        device (torch.Device): the device to test on
        batch_sizes (list[int], optional): list of batch sizes to test (Default value = (1, 16, 32, 64, 128)

    Returns:
        dict: dictionary of batch size to inference memory allocated

    """
    vram_allocated = {}

    for bs in sorted(batch_sizes, reverse=True):
        if input.shape[0] < bs:
            continue
        input = input[:bs]
        # reset statistics

        if args.compile_model:
            # force compilation first
            model(input)

        torch.cuda.reset_peak_memory_stats(device)

        amp_dtype = getattr(args, "amp_dtype", torch.float16)
        with torch.amp.autocast("cuda", dtype=amp_dtype) if args.eval_amp else nullcontext(), torch.no_grad():
            try:
                model(input)
                vram_allocated[bs] = max_mem_allocated(device, reset_max=True)
            except torch.cuda.OutOfMemoryError:
                pass
    return vram_allocated


def training_time(args, model, optim, scaler, data_loader, device, max_iters=200):
    from engine import setup_criteria_mixup

    criterion, _, mixup = setup_criteria_mixup(args)
    measure_steps = min(max_iters, len(data_loader))
    start_measure = int(measure_steps / 10)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i, (xs, ys) in tqdm(enumerate(data_loader), total=measure_steps, desc="Training time calculation"):
        if i == start_measure:
            starter.record()
        if i == measure_steps:
            break
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        if mixup:
            xs, ys = mixup(xs, ys)

        optim.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            preds = model(xs)
            loss = criterion(preds.transpose(1, -1), ys.transpose(1, -1) if len(ys.shape) > 1 else ys) + (
                model.get_internal_loss() if hasattr(model, "get_internal_loss") else model.module.get_internal_loss()
            )

        scaler(
            loss,
            optim,
            parameters=model.parameters(),
            clip_grad=args.max_grad_norm if args.max_grad_norm > 0.0 else None,
        )
    ender.record()
    torch.cuda.synchronize()

    return starter.elapsed_time(ender) / (measure_steps - start_measure)


def throughput(args, model, input, fixed_bs=None, iters=100, max_batch_size=None):
    """Calculate the throughput of a given model.

    Throughput is given for the biggest batch_size, that fits into memory. Images from input are repeated to get to this
    batch_size.
    Internally resets the max allocated memory, so only use this **after** *max_mem_allocated*.

    Args:
        args: training arguments; in particular set args.eval_amp
        model (torch.nn.Module): the model to analyze
        input (torch.Tensor): the batch of images to start with
        device (torch.cuda.device): the device to measure throughput with
        iters (int, optional): the number of iterations to test with (for more accurate numbers) (Default value = 100)

    Returns:
        tuple[int, int]: the optimal batch size and the throughput in images per second

    """
    # dev_properties = torch.cuda.get_device_properties(device)
    # total_mem = dev_properties.total_memory
    #
    # # reset max mem allocated
    # torch.cuda.reset_peak_memory_stats(device)
    #
    # n_ims = input.shape[0]
    # if n_ims > 4:
    #     n_ims = 4
    #     input = input[:4]
    # with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
    #     with torch.no_grad():
    #         try:
    #             model(input)
    #         except IndexError as e:
    #             logger.error(f"Index error {e} when calculating throughput. Might come from timm with amp.")
    #             return -1, -1
    # max_alloc = max_mem_allocated(device, reset_max=True)
    #
    # memory_allocated = {n_ims: max_alloc}
    #
    # if max_alloc <= (total_mem-250_000) // 2:
    #     input = torch.cat((input, input), dim=0)
    #     n_ims *= 2
    # else:
    #     input = input[:input.shape[0]//2]
    #     n_ims = n_ims // 2
    #
    # with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
    #     with torch.no_grad():
    #         model(input)
    # max_alloc = max_mem_allocated(device, reset_max=True)
    #
    # memory_allocated[n_ims] = max_alloc
    # pred_double = linear_regession(memory_allocated)(2*n_ims)
    #
    # torch.cuda.empty_cache()
    # while pred_double <= total_mem - 500_000_000:
    #     input = torch.cat((input, input), dim=0)
    #     n_ims = int(input.shape[0])
    #     try:
    #         with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
    #             with torch.no_grad():
    #                 model(input)
    #     except torch.cuda.OutOfMemoryError:
    #         break
    #     except RuntimeError as e:
    #         logger.error(f"RuntimeError '{e}' when calculating throughput (@{n_ims}). "
    #                       f"Might come from out- or input tensor size >2**31 (max int32_t).")
    #         logger.error(f"Stacktrace:\n"
    #                       f"{''.join(traceback.TracebackException.from_exception(e).format())}")
    #         break
    #     max_alloc = max_mem_allocated(device, reset_max=True)
    #     memory_allocated[n_ims] = max_alloc
    #     pred_double = linear_regession(memory_allocated)(2 * n_ims)
    #
    # reg_line = linear_regession(memory_allocated)
    # b = reg_line(0)
    # a = reg_line(1) - b
    #
    # assert input.shape[0] == n_ims, f"Found input of shape {input.shape}. Should have {n_ims} images."
    # test_bs = set([int(2 * n_ims / i) for i in range(1, 9)] +
    #               [int((total_mem - offset - b) / a) for offset in [250_000_000, 100_000_000, 0]])
    # test_bs = {2 ** math.floor(math.log2(bs)) for bs in test_bs if bs > 4}
    # test_bs = {bs - (bs % 16) for bs in test_bs}.union(test_bs)

    bs = min(1024, args.batch_size) if fixed_bs is None else fixed_bs
    # print(f"test batch sizes: {test_bs}")
    if input.shape[0] < bs:
        input = torch.cat((input for _ in range(int(bs / input.shape[0]) + 1)), dim=0)
    input = input[:bs]

    results = []
    n_decr = 0
    while fixed_bs is None and (max_batch_size is None or bs <= max_batch_size):
        # for bs in sorted(list(test_bs)):
        #     if input.shape[0] < bs:
        #         diff = bs - input.shape[0]
        #         input = torch.cat((input, input[:diff]), dim=0)
        #     else:
        #         input = input[:bs]
        #     n_ims = input.shape[0]
        print(f"thoughput calculation: test batch size {bs}")
        if args.cuda:
            try:
                tp = _measure_throughput_cuda(
                    model,
                    input,
                    iters,
                    args.eval_amp,
                    use_tqdm=args.tqdm,
                    amp_dtype=getattr(args, "amp_dtype", torch.float16),
                )
            except RuntimeError as e:
                if "canUse32BitIndexMath" in str(e):
                    print(f"throughput calculation: tensor too large @ {bs}")
                else:
                    print(f"throughput calculation: CUDA OOM @ {bs}")
                break
        else:
            tp = _measure_throughput_cpu(model, input, iters, use_tqdm=args.tqdm)
            print(f"used {_get_ram_usage()} MiB of {_get_ram_total()} MiB RAM")
        if len(results) > 1 and (tp < 0.98 * results[-1][1] or tp <= 0.95 * max(res_tp for _, res_tp in results)):
            n_decr += 1
        else:
            n_decr = 0
        print(f"decreasing trend for {n_decr} steps in a row")
        results.append((bs, tp))
        print(f"throughput calculation: throughput @ {bs} = {tp} images/second")
        if not args.cuda and _get_ram_usage() > 0.5 * _get_ram_total():
            print(f"throughput calculation: used more than 50% of total RAM @ {bs}; stopping further calculation")
            break
        if n_decr >= 2:
            print(
                "decreasing throughput trend for 3 sizes:"
                f" {' -> '.join([f'{tp_r:.2f} @ {bs_r}' for bs_r, tp_r in results[-3:]])}; stopping further calculation"
            )
            break
        if len(results) >= 2 and results[-1][1] < 0.5 * results[-2][1]:
            print(f"throughput calculation: throughput dropped below 50% of previous value @ {bs}; stopping now")
            break
        input = torch.cat((input, input), dim=0)
        bs *= 2
    # print(f"results {results}")
    if fixed_bs is None:
        top_bs, top_tp = max(results, key=lambda x: x[1]) if len(results) > 0 else (-1, -1)
    else:
        top_bs, top_tp = fixed_bs, None
        print(f"Calculating throughput for fixed batch size: {fixed_bs}")
    if top_tp is None or 10 * iters / (top_bs * top_tp) < 60 * 60:  # only measure again, if it takes less than 1 hour
        try:
            if args.cuda:
                top_tp = _measure_throughput_cuda(
                    model,
                    input[:top_bs],
                    iters * 10,
                    args.eval_amp,
                    use_tqdm=args.tqdm,
                    amp_dtype=getattr(args, "amp_dtype", torch.float16),
                )
            else:
                top_tp = _measure_throughput_cpu(model, input[:top_bs], iters * 10, use_tqdm=args.tqdm)
            print(f"Final througput @ {top_bs}: {top_tp}")
        except RuntimeError:
            pass
    return top_bs, top_tp


def _measure_throughput_cuda(model, input, iters=1000, eval_amp=False, use_tqdm=False, amp_dtype=torch.float16):
    """Measure the throughput of a PyTorch model on CUDA.

    Args:
        model (torch.nn.Module): The PyTorch model to measure throughput for.
        input (torch.Tensor): The input tensor of shape (batch_size, ...) for the model.
        iters (int, optional): The number of iterations to run to measure the throughput, by default 1000.
        eval_amp (bool, optional): Whether to evaluate using Automatic Mixed Precision (AMP) mode, by default False.
        use_tqdm (bool, optional): Show a progress bar using tqdm, by default False.

    Returns:
        float: The throughput in images per second.

    """
    # total_time = 0
    samples = []
    iterator = range(iters)
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"throughput calculation @ {input.shape[0]}", total=iters)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype) if eval_amp else nullcontext():
        for _ in iterator:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            __ = model(input)
            ender.record()
            torch.cuda.synchronize()
            # total_time += starter.elapsed_time(ender) / 1000  # ms -> s
            samples.append(starter.elapsed_time(ender) / 1000)  # ms -> s
    # return iters * input.shape[0]/total_time
    samples = samples[int(len(samples) / 10) :]
    return len(samples) * input.shape[0] / sum(samples)


def _measure_throughput_cpu(model, input, iters=1000, use_tqdm=False):
    """Measure the throughput of a PyTorch model on CPU.

    Args:
        model (torch.nn.Module): The PyTorch model to measure throughput for.
        input (torch.Tensor): The input tensor of shape (batch_size, ...) for the model.
        iters (int, optional): The number of iterations to run to measure the throughput, by default 1000.
        use_tqdm (bool, optional): Show a progress bar using tqdm, by default False.

    Returns:
        float: The throughput in images per second.

    """
    samples = []
    iterator = range(iters)
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"throughput calculation @ {input.shape[0]}", total=iters)
    for _ in iterator:
        with torch.no_grad():
            start = time()
            __ = model(input)
            end = time()
        samples.append(end - start)
    samples = samples[int(len(samples) / 10) :]
    return len(samples) * input.shape[0] / sum(samples)


def _get_ram_usage():
    """Return the RAM usage of this process in MiB."""
    # Get current process
    process = psutil.Process()
    # Get memory usage info in bytes
    mem_info = process.memory_info()
    # Convert bytes to MiB
    return mem_info.rss / (1024 * 1024)


def _get_ram_total():
    """Return the total RAM of this system in MiB."""
    return psutil.virtual_memory().total / 1024**2
