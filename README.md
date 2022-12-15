# Flair Models on NVIDIA Triton Server

Scripts to help deploy the Flair ner-english-fast model on Triton Server as a TorchScript model.

### Input

1. String to Integer:
   1. Sentence: `NVIDIA is founded by Jensen Huang , Chris Malachowsky and Curtis Priem .`
   2. Encoded Sentence:
      ```python
      array([ 78, 86, 73, 68, 73, 65, 32, 105, 115, 32, 102, 111, 117,
      110, 100, 101, 100, 32, 98, 121, 32, 74, 101, 110, 115, 101,
      110, 32, 72, 117, 97, 110, 103, 44, 32, 67, 104, 114, 105,
      115, 32, 77, 97, 108, 97, 99, 104, 111, 119, 115, 107, 121,
      32, 97, 110, 100, 32, 67, 117, 114, 116, 105, 115, 32, 80,
      114, 105, 101, 109, 46])
      ```

### Output

1. String to Integer:
   1. Tagged Sentence as Bytes
      ```python
      array([123,  39,  78,  86,  73,  68,  73,  65,  32, 105, 115,  32, 102,
       111, 117, 110, 100, 101, 100,  32,  98, 121,  32,  74, 101, 110,
       115, 101, 110,  32,  72, 117,  97, 110, 103,  32,  44,  32,  67,
       104, 114, 105, 115,  32,  77,  97, 108,  97,  99, 104, 111, 119,
       115, 107, 121,  32,  97, 110, 100,  32,  67, 117, 114, 116, 105,
       115,  32,  80, 114, 105, 101, 109,  32,  46,  39,  58,  32,  91,
       123,  39, 101, 110, 116, 105, 116, 121,  95, 103, 114, 111, 117,
       112,  39,  58,  32,  39,  79,  82,  71,  39,  44,  32,  39, 115,
       116,  97, 114, 116,  39,  58,  32,  48,  44,  32,  39, 119, 111,
       114, 100,  39,  58,  32,  39,  78,  86,  73,  68,  73,  65,  39,
        44,  32,  39, 101, 110, 100,  39,  58,  32,  54,  44,  32,  39,
       115,  99, 111, 114, 101,  39,  58,  32,  57,  57, 125,  44,  32,
       123,  39, 101, 110, 116, 105, 116, 121,  95, 103, 114, 111, 117,
       112,  39,  58,  32,  39,  80,  69,  82,  39,  44,  32,  39, 115,
       116,  97, 114, 116,  39,  58,  32,  50,  49,  44,  32,  39, 119,
       111, 114, 100,  39,  58,  32,  39,  74, 101, 110, 115, 101, 110,
        32,  72, 117,  97, 110, 103,  39,  44,  32,  39, 101, 110, 100,
        39,  58,  32,  51,  51,  44,  32,  39, 115,  99, 111, 114, 101,
        39,  58,  32,  57,  57, 125,  44,  32, 123,  39, 101, 110, 116,
       105, 116, 121,  95, 103, 114, 111, 117, 112,  39,  58,  32,  39,
        80,  69,  82,  39,  44,  32,  39, 115, 116,  97, 114, 116,  39,
        58,  32,  51,  53,  44,  32,  39, 119, 111, 114, 100,  39,  58,
        32,  39,  67, 104, 114, 105, 115,  32,  77,  97, 108,  97,  99,
       104, 111, 119, 115, 107, 121,  39,  44,  32,  39, 101, 110, 100,
        39,  58,  32,  53,  50,  44,  32,  39, 115,  99, 111, 114, 101,
        39,  58,  32,  57,  57, 125,  44,  32, 123,  39, 101, 110, 116,
       105, 116, 121,  95, 103, 114, 111, 117, 112,  39,  58,  32,  39,
        80,  69,  82,  39,  44,  32,  39, 115, 116,  97, 114, 116,  39,
        58,  32,  53,  55,  44,  32,  39, 119, 111, 114, 100,  39,  58,
        32,  39,  67, 117, 114, 116, 105, 115,  32,  80, 114, 105, 101,
       109,  39,  44,  32,  39, 101, 110, 100,  39,  58,  32,  54,  57,
        44,  32,  39, 115,  99, 111, 114, 101,  39,  58,  32,  57,  57,
       125,  93, 125])
      ```
   2. Decoded:
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

## Model Conversion to TorchScript

The flair models are a composition of multiple models which complete embedding and tagging of strings. The default method modifies a `Sentence` inplace. Serving on Triton will convert any string-as-bytes input to the appropriate tokenization and embeding used in the original flair ner-fast model. You can convert the original meta-model hosted on HuggingFace by running:

```sh
docker compose up ner-english-fast
```

and from within the Jupyter Lab terminal, `python3 flair_model_surgery.py`.

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
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Using "time_windows" mode for stabilization
  Measurement window: 30000 msec
  Using asynchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 4
  Pass [1] throughput: 92.8314 infer/sec. p95 latency: 44717 usec
  Pass [2] throughput: 93.9971 infer/sec. p95 latency: 44342 usec
  Pass [3] throughput: 92.8868 infer/sec. p95 latency: 44900 usec
  Client:
    Request count: 10070
    Throughput: 93.2384 infer/sec
    p50 latency: 42808 usec
    p90 latency: 44314 usec
    p95 latency: 44660 usec
    p99 latency: 45685 usec
    Avg gRPC time: 42844 usec (marshal 2 usec + response wait 42842 usec + unmarshal 0 usec)
  Server:
    Inference count: 10070
    Execution count: 10070
    Successful request count: 10070
    Avg request latency: 42650 usec (overhead 29 usec + queue 11012 usec + compute 31609 usec)

  Composing models:
  flair-ner-english-fast, version:
      Inference count: 10072
      Execution count: 10072
      Successful request count: 10072
      Avg request latency: 4271 usec (overhead 15 usec + queue 751 usec + compute input 20 usec + compute infer 2223 usec + compute output 1261 usec)

  flair-ner-english-fast-tokenization, version:
      Inference count: 10073
      Execution count: 10073
      Successful request count: 10073
      Avg request latency: 6724 usec (overhead 4 usec + queue 40 usec + compute input 7 usec + compute infer 6644 usec + compute output 29 usec)

  flair-ner-english-fast-viterbi-decoder, version:
      Inference count: 10070
      Execution count: 10070
      Successful request count: 10070
      Avg request latency: 31653 usec (overhead 8 usec + queue 10221 usec + compute input 45 usec + compute infer 21327 usec + compute output 52 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 4, throughput: 93.2384 infer/sec, latency 44660 usec
```
