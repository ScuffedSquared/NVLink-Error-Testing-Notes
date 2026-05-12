# NVLink-Error-Testing-Notes
NVLink Error Testing Notes for the https://www.ebay.com/itm/298159550124 

# NVLink Error Testing Notes

## Test 1

### Action: Reset NVLink Error Counters

I reset the NVLink error counters before testing.

```bash
sudo nvidia-smi nvlink -re
```

```text
GPU 0: Tesla V100-SXM2-32GB-LS (UUID: GPU-5261e3df-82f6-67b4-9953-4e63527e0e9d)
         All links (0-5): error counters reset

GPU 2: Tesla V100-SXM2-32GB-32GB-LS (UUID: GPU-d9eeee97-c180-66de-23e0-d995387934dc)
         All links (0-5): error counters reset
```

### Action: Check NVLink Errors After Reset

I checked the current NVLink error counters.

```bash
sudo nvidia-smi nvlink -e
```

```text
GPU 0: Tesla V100-SXM2-32GB-LS (UUID: GPU-5261e3df-82f6-67b4-9953-4e63527e0e9d)
         Links 0-5: No errors

GPU 2: Tesla V100-SXM2-32GB-LS (UUID: GPU-d9eeee97-c180-66de-23e0-d995387934dc)
         Link 0: CRC Errors: 8814
         Links 1-5: No errors
```

### Change Observed

GPU 2 Link 0 already had CRC errors after the reset/check sequence.

## Error Accumulation

The CRC errors continued accumulating until they reached `65535`.

```bash
sudo nvidia-smi nvlink -e
```

```text
GPU 0: Tesla V100-SXM2-32GB-LS (UUID: GPU-5261e3df-82f6-67b4-9953-4e63527e0e9d)
         Links 0-5: CRC Errors: 0

GPU 2: Tesla V100-SXM2-32GB-LS (UUID: GPU-d9eeee97-c180-66de-23e0-d995387934dc)
         Link 0: CRC Errors: 65535
         Links 1-5: CRC Errors: 0
```

### Change Observed

GPU 2 Link 0 reached the maximum displayed CRC error count.

## Action: Run Containerized NCCL Stress Test

I launched the vLLM containerized NCCL stress test.

```bash
sudo docker exec -e NCCL_DEBUG=INFO -e NCCL_P2P_LEVEL=NVL ... 1cat-vllm-sm70 bash -lc 'cat > /tmp/nccl_stress_small.py ...'
```

### Action: Monitor Initialization and Progress

I tracked NCCL initialization and timing.

```text
[NCCL Initialization]
- Using network Socket
- Rank 0 and Rank 1 communication initialized

[Timing Progress]
iter=0  elapsed=0.20s
...
iter=975 elapsed=1.05s

Result:
Test completed without crashing.
```

### Action: Check NVLink Errors After Stress Test

I checked the NVLink counters again after the stress test.

```bash
sudo nvidia-smi nvlink -e
```

```text
GPU 0: Tesla V100-SXM2-32GB-LS (UUID: GPU-5261e3df-82f6-67b4-9953-4e63527e0e9d)
         Link 3: Replay Errors: 2353
         Links 0-2, 4-5: Replay/Recovery/CRC Errors: 0

GPU 2: Tesla V100-SXM2-32GB-LS (UUID: GPU-d9eeee97-c180-66de-23e0-d995387934dc)
         Link 0: Replay Errors: 148
         Link 0: CRC Errors: 65535
         Links 1-5: Replay/Recovery/CRC Errors: 0
```

### Change Observed

After the stress test, GPU 0 Link 3 started showing replay errors. GPU 2 Link 0 remained at `65535` CRC errors.

I expect GPU 0 Link 3 errors to continue rising if testing continues, eventually reaching `65535` and crashing the system.

## Hardware Swap

### Action: Replace GPUs

I swapped GPU 0 and GPU 2 with replacement GPUs.

```text
Old GPU 0 UUID:
GPU-5261e3df-82f6-67b4-9953-4e63527e0e9d

New GPU 0 UUID:
GPU-bd0738fc-434f-4a42-28e0-d699cb37f301

Old GPU 2 UUID:
GPU-d9eeee97-c180-66de-23e0-d995387934dc

New GPU 2 UUID:
GPU-926a4a2e-2909-4d8a-fb23-6d90ed5bcb14
```

### Action: Inspect Pins

I checked the GPU pins and the NVLink interconnect board pins.

```text
Result:
No debris found.
No bent pins found.
No visible physical issues found.
```

## Test 2

### Action: Check NVLink Errors With Replacement GPUs

I checked NVLink errors again after installing the replacement GPUs.

```bash
sudo nvidia-smi nvlink -e
```

```text
GPU 0: Tesla V100-SXM2-32GB-LS (UUID: GPU-bd0738fc-434f-4a42-28e0-d699cb37f301)
         Links 0-5: Replay/Recovery/CRC Errors: 0

GPU 2: Tesla V100-SXM2-32GB-LS (UUID: GPU-926a4a2e-2909-4d8a-fb23-6d90ed5bcb14)
         Link 0: CRC Errors: 2805
         Links 1-5: Replay/Recovery/CRC Errors: 0
```

### Change Observed

The replacement GPU in the GPU 2 position still showed CRC errors on Link 0.

## Action: Run Containerized Stress Test Again

I ran the same containerized NCCL stress test again.

```bash
sudo docker exec \
  -e NCCL_DEBUG=INFO \
  -e NCCL_P2P_LEVEL=NVL \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  1cat-vllm-sm70 \
  bash -lc 'cat > /tmp/nccl_stress_small.py <<'"'"'PY'"'"'
import os, time, torch, torch.distributed as dist

rank = int(os.environ[["LOCAL_RANK"]])
torch.cuda.set_device(rank)

dist.init_process_group("nccl")

x = torch.ones((32, 1024, 1024), device="cuda", dtype=torch.float16)
torch.cuda.synchronize()

start = time.time()

for i in range(1000):
    dist.all_reduce(x)
    if i % 25 == 0:
        torch.cuda.synchronize()
        if rank == 0:
            print(f"iter={i} elapsed={time.time() - start:.2f}s", flush=True)

dist.destroy_process_group()
PY
torchrun --standalone --nproc_per_node=2 /tmp/nccl_stress_small.py'
```

```text
NCCL INFO Bootstrap: Using eth0:172.29.0.2<0>
NCCL INFO cudaDriverVersion 13000
NCCL INFO NCCL version 2.27.5+cuda12.9
NCCL INFO NCCL_P2P_LEVEL set by environment to NVL
NCCL INFO 12 coll channels, 0 nvls channels, 16 p2p channels
NCCL INFO Channel 00/0 : 0 -> 1 via P2P/CUMEM
NCCL INFO Channel 00/0 : 1 -> 0 via P2P/CUMEM
NCCL INFO Connected all rings

iter=0 elapsed=0.19s
iter=25 elapsed=0.21s
...
iter=975 elapsed=1.04s

NCCL INFO Destroy COMPLETE
```

### Change Observed

The stress test completed, but it created additional NVLink errors.

## Action: Check NVLink Errors After Second Stress Test

I checked the NVLink counters again.

```bash
sudo nvidia-smi nvlink -e
```

```text
GPU 0: Tesla V100-SXM2-32GB-LS (UUID: GPU-bd0738fc-434f-4a42-28e0-d699cb37f301)
         Link 3: Replay Errors: 104
         Links 0-2, 4-5: Replay/Recovery/CRC Errors: 0

GPU 2: Tesla V100-SXM2-32GB-LS (UUID: GPU-926a4a2e-2909-4d8a-fb23-6d90ed5bcb14)
         Link 0: Replay Errors: 2
         Link 0: CRC Errors: 45068
         Links 1-5: Replay/Recovery/CRC Errors: 0
```

### Change Observed

GPU 0 Link 3 gained replay errors again.

GPU 2 Link 0 CRC errors increased from `2805` to `45068`.

## Summary

After replacing GPU 0 and GPU 2, the same NVLink error pattern continued.

```text
GPU 0:
- Link 3 develops replay errors during stress testing.

GPU 2:
- Link 0 develops CRC errors.
- CRC errors increase rapidly under load.
- Link 0 reached 65535 in Test 1.
- Link 0 increased from 2805 to 45068 in Test 2.

All other links:
- No significant errors observed.
```

The issue appears to follow the GPU slot/link path rather than the original GPUs, because replacement GPUs produced the same error behavior.
