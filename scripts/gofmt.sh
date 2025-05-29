#!/bin/bash

mismatched_files=$(gofmt -l .)

if [[ -n "$mismatched_files" ]]; then
    echo "🚫 The following files are not formatted correctly (via gofmt):"
    echo "$mismatched_files" | sed 's/^/  - /'
    exit 1
else
    echo "✅ All Go files are properly formatted."
fi
