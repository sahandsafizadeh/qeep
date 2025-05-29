#!/bin/bash
set -e

GO_BUILD_TAGS=$1

if [ -n "$GO_BUILD_TAGS" ]; then
    gosec -tags "$GO_BUILD_TAGS" ./...
else
    gosec ./...
fi
