# Flair Models on NVIDIA Triton Server

Scripts to help deploy the Flair ner-english-fast model on Triton Server as a TorchScript model.

### Input

1. Sentence(s)

```bash
NVIDIA is founded by Jensen Huang , Chris Malachowsky and Curtis Priem .
```

### Output

1. Decoded

```json
{
  "NVIDIA is founded by Jensen Huang , Chris Malachowsky and Curtis Priem .": [
    {
      "entity_group": "ORG",
      "start": 0,
      "word": "NVIDIA",
      "end": 6,
      "score": 99
    },
    {
      "entity_group": "PER",
      "start": 21,
      "word": "Jensen Huang",
      "end": 33,
      "score": 99
    },
    {
      "entity_group": "PER",
      "start": 35,
      "word": "Chris Malachowsky",
      "end": 52,
      "score": 99
    },
    {
      "entity_group": "PER",
      "start": 57,
      "word": "Curtis Priem",
      "end": 69,
      "score": 99
    }
  ]
}
```

## Model download

You'll need download the original model from flair and place it in workspace/triton-models/ner-english-fast/1/model.bin

## Launch Triton Server

The prior model conversion step will place the `model.pt` file in the `workspace/triton-models/flair-ner-english-fast/1` folder. This will then be bound to the container for serving and will be the model_repo. The default configuration assumes serving is done on an NVIDIA GPU but this could be modified here: `workspace/triton-models/flair-ner-english-fast/config.pbtxt`

```sh
docker compose up triton-server
```

## Performance

There is a script which utilized NVIDIA"s `perf_analyzer` tool which can quickly report details about latency for your model set up.

```sh
docker compose up triton-client
```

```sh
docker compose run perf-analyzer
```

Below are some preliminary results to serve as a simple benchmark to be improved upon.

```sh
*** Measurement Settings ***
  Batch size: 32
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 30000 msec
  Latency limit: 0 msec
  Concurrency limit: 64 concurrent requests
  Using asynchronous calls for inference
  Stabilizing using p99 latency

Request concurrency: 32
  Client: 
    Request count: 6945
    Throughput: 2057.75 infer/sec
    p50 latency: 498487 usec
    p90 latency: 547042 usec
    p95 latency: 561698 usec
    p99 latency: 590426 usec
    Avg gRPC time: 496396 usec ((un)marshal request/response 3 usec + response wait 496393 usec)
  Server: 
    Inference count: 222240
    Execution count: 6945
    Successful request count: 6945
    Avg request latency: 496099 usec (overhead 7 usec + queue 465015 usec + compute input 11 usec + compute infer 31006 usec + compute output 59 usec)

Request concurrency: 48
  Client: 
    Request count: 6945
    Throughput: 2057.75 infer/sec
    p50 latency: 747909 usec
    p90 latency: 800052 usec
    p95 latency: 814290 usec
    p99 latency: 835362 usec
    Avg gRPC time: 744492 usec ((un)marshal request/response 3 usec + response wait 744489 usec)
  Server: 
    Inference count: 222240
    Execution count: 6945
    Successful request count: 6945
    Avg request latency: 744258 usec (overhead 6 usec + queue 713167 usec + compute input 11 usec + compute infer 31013 usec + compute output 60 usec)

Request concurrency: 64
  Client: 
    Request count: 6912
    Throughput: 2047.97 infer/sec
    p50 latency: 998946 usec
    p90 latency: 1057445 usec
    p95 latency: 1078742 usec
    p99 latency: 1121684 usec
    Avg gRPC time: 997158 usec ((un)marshal request/response 3 usec + response wait 997155 usec)
  Server: 
    Inference count: 221184
    Execution count: 6912
    Successful request count: 6912
    Avg request latency: 996923 usec (overhead 7 usec + queue 965683 usec + compute input 11 usec + compute infer 31161 usec + compute output 60 usec)

Inferences/Second vs. Client p99 Batch Latency
Concurrency: 32, throughput: 2057.75 infer/sec, latency 590426 usec
Concurrency: 48, throughput: 2057.75 infer/sec, latency 835362 usec
Concurrency: 64, throughput: 2047.97 infer/sec, latency 1121684 usec
```
