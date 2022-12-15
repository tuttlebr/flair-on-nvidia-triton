#!/bin/bash

perf_analyzer \
    -m flair-ner-english-fast-ensemble \
    --async \
    --percentile=95 \
    --concurrency-range 4 \
    --input-data /data/perf_analyzer_data.json \
    --shape sentence_bytes:70 \
    -u 172.25.4.42:8001 \
    -i grpc \
    --measurement-interval 30000 \
    -v