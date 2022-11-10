# Flair Models on NVIDIA Triton Server

## Model Conversion to TorchScript

The flair models are a composition of multiple models which complete embedding and tagging of strings. The default method modifies a `Sentence` inplace. Serving on Triton will convert any string-as-bytes input to the appropriate tokenization and embeding used in the original flair ner-fast model. You can convert the original meta-model hosted on HuggingFace by running:

```sh
docker compose up ner-english-fast
```

## Launch Triton Server

The prior model conversion step will place the `model.pt` file in the `workspace/triton-models/flair-ner-english-fast/1` folder. This will then be bound to the container for serving and will be the model_repo. The default configuration assumes serving is done on an NVIDIA GPU but this could be modified here: `workspace/triton-models/flair-ner-english-fast/config.pbtxt`

## Performance

There is a script which utilized NVIDIA's `perf_analyzer` tool which can quickly report details about latency for your model set up.
```sh
docker compose up triton-client
```

Below are some preliminary results to serve as a simple benchmark to be improved upon. 

| Name                   | Platform         | Inputs | Outputs | Batch | Status |
| ---------------------- | ---------------- | ------ | ------- | ----- | ------ |
| flair-ner-english-fast | pytorch_libtorch | 1      | 1       | 64    | OK     |

```sh
*** Measurement Settings ***
  Batch size: 64
  Using "time_windows" mode for stabilization
  Measurement window: 30000 msec
  Latency limit: 0 msec
  Concurrency limit: 96 concurrent requests
  Using asynchronous calls for inference
  Stabilizing using average latency

Request concurrency: 24
  Client:
    Request count: 3313551
    Throughput: 7.06891e+06 infer/sec
    Avg latency: 210 usec (standard deviation 73 usec)
    p50 latency: 199 usec
    p90 latency: 288 usec
    p95 latency: 323 usec
    p99 latency: 411 usec
    Avg gRPC time: 207 usec ((un)marshal request/response 1 usec + response wait 206 usec)
  Server:
    Inference count: 254398528
    Execution count: 3974977
    Successful request count: 3974976
    Avg request latency: 61 usec (overhead 12 usec + queue 23 usec + compute input 18 usec + compute infer 8 usec + compute output 0 usec)

Request concurrency: 48
  Client:
    Request count: 3760161
    Throughput: 8.02168e+06 infer/sec
    Avg latency: 374 usec (standard deviation 214 usec)
    p50 latency: 329 usec
    p90 latency: 553 usec
    p95 latency: 704 usec
    p99 latency: 1242 usec
    Avg gRPC time: 371 usec ((un)marshal request/response 1 usec + response wait 370 usec)
  Server:
    Inference count: 288757632
    Execution count: 4511839
    Successful request count: 4511839
    Avg request latency: 112 usec (overhead 13 usec + queue 72 usec + compute input 18 usec + compute infer 8 usec + compute output 0 usec)

Request concurrency: 72
  Client:
    Request count: 3864362
    Throughput: 8.24397e+06 infer/sec
    Avg latency: 549 usec (standard deviation 433 usec)
    p50 latency: 450 usec
    p90 latency: 783 usec
    p95 latency: 1189 usec
    p99 latency: 2517 usec
    Avg gRPC time: 547 usec ((un)marshal request/response 1 usec + response wait 546 usec)
  Server:
    Inference count: 296704896
    Execution count: 4636014
    Successful request count: 4636014
    Avg request latency: 176 usec (overhead 14 usec + queue 134 usec + compute input 19 usec + compute infer 8 usec + compute output 0 usec)

Request concurrency: 96
  Client:
    Request count: 3901155
    Throughput: 8.32246e+06 infer/sec
    Avg latency: 727 usec (standard deviation 687 usec)
    p50 latency: 559 usec
    p90 latency: 980 usec
    p95 latency: 1820 usec
    p99 latency: 4038 usec
    Avg gRPC time: 726 usec ((un)marshal request/response 1 usec + response wait 725 usec)
  Server:
    Inference count: 299209920
    Execution count: 4675155
    Successful request count: 4675155
    Avg request latency: 231 usec (overhead 14 usec + queue 188 usec + compute input 20 usec + compute infer 8 usec + compute output 0 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 24, throughput: 7.06891e+06 infer/sec, latency 210 usec
Concurrency: 48, throughput: 8.02168e+06 infer/sec, latency 374 usec
Concurrency: 72, throughput: 8.24397e+06 infer/sec, latency 549 usec
Concurrency: 96, throughput: 8.32246e+06 infer/sec, latency 727 usec
---------------------------------------------------------------------------------------
```
