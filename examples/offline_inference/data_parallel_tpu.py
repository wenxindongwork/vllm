# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TPU-specific Data Parallel Inference Example

Usage:
Single node (8 TPU chips):
    python examples/offline_inference/data_parallel_tpu.py \
            --model="meta-llama/Llama-3.1-8B-Instruct" \
            --dp-size=2 \
            --tp-size=4

This example demonstrates how to run data parallel inference on TPUs.
Each DP rank will use 4 TPU chips (specified by tp-size), and 2 DP ranks
will run in parallel for a total of 8 TPU chips.

TPU Environment Variables Set:
- TPU_CHIPS_PER_PROCESS_BOUNDS: Controls chip allocation per process
- TPU_PROCESS_BOUNDS: Process boundary configuration  
- TPU_VISIBLE_CHIPS: Which specific chips each process can see
"""

import importlib.util
import multiprocessing
import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


os.environ['SKIP_JAX_PRECOMPILE'] = '1'
# os.environ['JAX_RANDOM_WEIGHTS'] = '1'


# Check if TPU-related modules are available
TPU_AVAILABLE = importlib.util.find_spec("jax") is not None

# Environment variable constants for TPU
TPU_CHIPS_PER_PROCESS_BOUNDS = "TPU_CHIPS_PER_PROCESS_BOUNDS"
TPU_PROCESS_BOUNDS = "TPU_PROCESS_BOUNDS"
TPU_VISIBLE_CHIPS = "TPU_VISIBLE_CHIPS"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="TPU Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/wenxindong_google_com/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=2, help="Data parallel size")
    parser.add_argument(
        "--tp-size", 
        type=int, 
        default=4, 
        help="Tensor parallel size (TPU chips per DP rank)"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode execution."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code."
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help=("Maximum number of sequences to be processed in a single iteration."),
    )
    parser.add_argument(
        "--total-chips",
        type=int,
        default=8,
        help="Total number of TPU chips available",
    )
    return parser.parse_args()


def tpu_worker_process(
    process_id: int,
    visible_chips: str,
    model: str,
    dp_size: int,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
    tp_size: int,
    enforce_eager: bool,
    trust_remote_code: bool,
    max_num_seqs: int,
) -> None:
    """Worker function that runs in each TPU process."""
    
    def log(message: str):
        print(f"[DP-{global_dp_rank}] {message}")
    
    log(f"Starting TPU process {process_id}")
    
    # Set vLLM DP environment variables
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    
    
    # Set TPU environment variables to limit visible chips
    os.environ[TPU_CHIPS_PER_PROCESS_BOUNDS] = "1,4,1"
    os.environ[TPU_PROCESS_BOUNDS] = "1,1,1"
    # os.environ[TPU_VISIBLE_CHIPS] = visible_chips
    
#     os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,2,1"
# os.environ["TPU_PROCESS_BOUNDS"] = "2,1,1"
# os.environ["TPU_PROCESS_ADDRESSES"] = "localhost:8476,localhost:8477"
# os.environ["TPU_VISIBLE_DEVICES"] = "0,1" # "2,3"
# os.environ["TPU_PROCESS_PORT"] = "8476" # "8477"
# os.environ["CLOUD_TPU_TASK_ID"] = "0" # "1"



    # log(f"Set {TPU_VISIBLE_CHIPS} to: {visible_chips}")
    log(f"Set {TPU_CHIPS_PER_PROCESS_BOUNDS} to: 1,{tp_size},1")

    # Sample prompts - distribute across DP ranks
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Artificial intelligence will",
        "The best programming language is",
        "Climate change is",
        "The universe consists of",
    ] * 10  # 80 prompts total

    # Distribute prompts among DP ranks
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def start(rank):
        return rank * floor + min(rank, remainder)

    rank_prompts = prompts[start(global_dp_rank) : start(global_dp_rank + 1)]
    if len(rank_prompts) == 0:
        rank_prompts = ["Placeholder"]
    
    log(f"Processing {len(rank_prompts)} prompts")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.8, 
        top_p=0.95, 
        max_tokens=32
    )

    try:
        # Create LLM instance for TPU
        log("Initializing LLM...")
        llm = LLM(
            model=model,
            tensor_parallel_size=tp_size,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            max_num_seqs=max_num_seqs,
            # Note: No gpu_memory_utilization for TPU
        )
        
        log("Starting inference...")
        outputs = llm.generate(rank_prompts, sampling_params)
        
        # Print results
        log("Inference completed. Results:")
        for i, output in enumerate(outputs[:3]):  # Print first 3 results
            prompt = output.prompt
            generated_text = output.outputs[0].text
            log(f"  Prompt: {prompt!r}")
            log(f"  Generated: {generated_text!r}")
            log("")
        
        if len(outputs) > 3:
            log(f"... and {len(outputs) - 3} more outputs")
            
    except Exception as e:
        log(f"Error during inference: {e}")
        raise
    
    # Give engines time to clean up
    sleep(1)
    log(f"Process {process_id} completed successfully")


def main():
    """Main function that spawns multiple TPU processes for data parallelism."""
    
    if not TPU_AVAILABLE:
        raise RuntimeError(
            "JAX is not available. Please install JAX with TPU support."
        )
    
    args = parse_args()
    
    print("Starting TPU Data Parallel Inference")
    print(f"Model: {args.model}")
    print(f"Data Parallel Size: {args.dp_size}")
    print(f"Tensor Parallel Size (chips per DP rank): {args.tp_size}")
    print(f"Total TPU chips: {args.total_chips}")
    
    # Validate configuration
    total_chips_needed = args.dp_size * args.tp_size
    if total_chips_needed > args.total_chips:
        raise ValueError(
            f"Not enough TPU chips. Need {total_chips_needed} but only "
            f"{args.total_chips} available."
        )
    
    # Setup for single node
    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()
    
    print(f"Using master address: {dp_master_ip}:{dp_master_port}")
    
    # Calculate chip assignments for each DP rank
    processes = []
    for local_dp_rank in range(args.dp_size):
        global_dp_rank = local_dp_rank  # Single node case
        
        # Calculate which chips this DP rank should use
        start_chip = local_dp_rank * args.tp_size
        end_chip = start_chip + args.tp_size - 1
        visible_chips = ",".join(str(i) for i in range(start_chip, end_chip + 1))
        
        print(f"Starting DP rank {global_dp_rank} with TPU chips {visible_chips}")
        
        # Create process using fork context (required for TPU)
        process = multiprocessing.get_context("fork").Process(
            target=tpu_worker_process,
            args=(
                local_dp_rank + 1,  # process_id for logging
                visible_chips,
                args.model,
                args.dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                args.tp_size,
                args.enforce_eager,
                args.trust_remote_code,
                args.max_num_seqs,
            ),
        )
        
        process.start()
        processes.append(process)
    
    # Wait for all processes to complete
    exit_code = 0
    for i, process in enumerate(processes):
        process.join(timeout=300)  # 5 minute timeout
        if process.exitcode is None:
            print(f"Killing process {i} that didn't complete within 5 minutes.")
            process.kill()
            exit_code = 1
        elif process.exitcode != 0:
            print(f"Process {i} exited with code {process.exitcode}")
            exit_code = process.exitcode
    
    if exit_code == 0:
        print("All processes completed successfully!")
    else:
        print(f"Some processes failed. Exit code: {exit_code}")
    
    exit(exit_code)


if __name__ == "__main__":
    main()
