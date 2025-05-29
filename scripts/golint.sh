#!/bin/bash
set -e

GO_BUILD_TAGS=$1

if [ -n "$GO_BUILD_TAGS" ]; then
    golangci-lint run --build-tags "$GO_BUILD_TAGS" ./...
else
    golangci-lint run ./...
fi
