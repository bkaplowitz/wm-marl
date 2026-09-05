# Production-size recurrent training gate

Both three-update gates passed on NVIDIA A100-SXM4-80GB GPUs using the frozen
corrected 2s3z checkpoint at environment step 10530. The exact saved production
batch/context/model/PPO settings were preserved. Recurrent treatment used
H2/H4/H5, eight sampled anchors, scale 0.1, consumer KL 0. Source commit:
`da98e2d683cfa28af9a261a56489a058a2567d4d`; canonical package SHA256:
`a06ed84235620d433a2030189e7017d05393d32a5b7247325cb37a046d64e7bc`.

| Measurement | Recurrent | Auxiliary disabled |
| --- | ---: | ---: |
| Initialization | 81.69 s | 74.03 s |
| First update, including compilation | 667.39 s | 526.56 s |
| Subsequent update 1 | 0.6697 s | 0.6394 s |
| Subsequent update 2 | 0.6694 s | 0.6342 s |
| Peak live JAX allocation | 19.53 GiB | 13.88 GiB |
| Sampled process allocation peak | 21,080 MiB | 21,068 MiB |

The recurrent update took 5.15% longer over the two steady measurements. Both
processes reserved similar allocator pools despite different live allocation
peaks. These are a bounded executability/cost check, not a throughput benchmark
or evidence of improved policy performance.

Every update had finite parameters, optimizer states and metrics, active PPO,
and no skipped local/joint world updates. Recurrent H5 valid-head counts were
40, 35, 40. Both checkpoint counters advanced from 1157 to 1160 updates; actor-action
counters stayed 10530. All preserved checkpoint/config/replay source files were
hash-identical after execution. No environment collection or checkpoint save
occurred. The replay sampler and checkpoint seeds were restored identically;
the script used the normal prefetched replay transport, without an additional
pre-materialized batch artifact.

Recurrent PID 42001 and control PID 43000 exited. Final direct `nvidia-smi` queries
confirmed no compute processes on assigned GPUs 4 and 1 before releasing them to
the experiment launcher. GPU 0 was untouched by these gates.

`summary.json` contains the essential results. Each mode subdirectory contains
complete scalar metrics, config, runtime/source/input provenance and memory
records. Launch records and stdout logs are alongside them. Remote originals:
`/workspace/majepa_self_fed_gate_20260905/` on the assigned RunPod host.
