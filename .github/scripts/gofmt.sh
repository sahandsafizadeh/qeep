#!/bin/bash

mismatched_files=$(gofmt -l .)

if [ -n "$mismatched_files" ]; then
    echo "The following files are not formatted correctly via 'gofmt':"
    echo "$mismatched_files"
    exit 1
fi
