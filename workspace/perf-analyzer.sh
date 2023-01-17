#!/bin/bash
MODEL=ner-english-fast
ENDPOINT=172.22.4.42
MAX_THREADS=32
MIN_CONCURRENCY=32
MAX_CONCURRENCY=64
STEP_CONCURRENCY=16
BATCH_SIZE=32

perf_analyzer \
    -m ${MODEL} \
    -a \
    -i grpc \
    -u ${ENDPOINT}:8001 \
    --string-data "NVIDIA is founded by Jensen Huang, Chris Malachowsky and Curtis Priem." \
    --shape INPUT_0:1 \
    --percentile 95 \
    --max-threads ${MAX_THREADS} \
    --request-distribution constant \
    --measurement-interval 30000 \
    --percentile=99 \
    --concurrency-range ${MIN_CONCURRENCY}:${MAX_CONCURRENCY}:${STEP_CONCURRENCY} \
    -b ${BATCH_SIZE}