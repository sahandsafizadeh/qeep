#!/bin/bash
set -e

GO_BUILD_TAGS=$1

go build -tags="$GO_BUILD_TAGS" ./...
