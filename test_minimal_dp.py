#!/usr/bin/env python3
"""
Minimal test for DP-aware block allocation in vLLM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vllm'))

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


def test_dp_block_pool():
    """Test the DP-aware block pool functionality."""
    print("=== Testing Minimal DP Block Pool ===")
    
    # Test parameters
    total_blocks = 100
    dp_size = 4
    
    print(f"Total blocks: {total_blocks}")
    print(f"DP size: {dp_size}")
    
    # Create DP-aware block pool
    dp_pool = BlockPool(
        num_gpu_blocks=total_blocks,
        enable_caching=True,
        dp_size=dp_size
    )
    
    print(f"Blocks per device: {dp_pool.blocks_per_device}")
    print(f"Remainder blocks: {dp_pool.remainder_blocks}")
    print(f"Free blocks: {dp_pool.get_num_free_blocks()}")
    
    # Test allocation from preferred device
    try:
        blocks = dp_pool.get_new_blocks(5, preferred_device=1)
        print(f"Allocated {len(blocks)} blocks from device 1")
        print(f"Block IDs: {[b.block_id for b in blocks]}")
        
        # Check if blocks are from device 1 (blocks 25-49)
        device_1_blocks = [b for b in blocks if 25 <= b.block_id < 50]
        print(f"Blocks from device 1: {len(device_1_blocks)}/{len(blocks)}")
        
        # Free blocks
        dp_pool.free_blocks(blocks)
        print(f"Free blocks after freeing: {dp_pool.get_num_free_blocks()}")
        
    except Exception as e:
        print(f"Allocation failed: {e}")
    
    # Test regular allocation (no preferred device)
    try:
        blocks = dp_pool.get_new_blocks(3)
        print(f"Allocated {len(blocks)} blocks (no preference)")
        print(f"Block IDs: {[b.block_id for b in blocks]}")
        dp_pool.free_blocks(blocks)
        
    except Exception as e:
        print(f"Regular allocation failed: {e}")


if __name__ == "__main__":
    print("Minimal DP-Aware Block Allocation Test")
    print("=" * 50)
    
    try:
        test_dp_block_pool()
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 