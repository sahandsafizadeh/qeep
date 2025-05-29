#!/bin/bash
set -e

# coverage configs
GO_BUILD_TAGS=$1
MIN_COVERAGE_PERCENT=90.0
EXCLUDED_COVERAGE_PATH='examples'

# output paths
COVERAGE_OUT=coverage.out
PROFILE_OUT=profile.out

# ANSI color codes
NC='\033[0m'
RED='\033[0;31m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'

echo -e "${CYAN}▶ Running tests with coverage...${NC}"
echo -e "----------------------------------------"

cover_pkgs=$(go list -tags="$GO_BUILD_TAGS" ./... | grep -v "$EXCLUDED_COVERAGE_PATH" | tr '\n' ',')
go test -tags="$GO_BUILD_TAGS" -coverpkg="$cover_pkgs" -coverprofile="$PROFILE_OUT" ./... || {
    echo -e "${RED}❌ Tests failed.${NC}"
    exit 1
}
go tool cover -func="$PROFILE_OUT" >"$COVERAGE_OUT"

rm "$PROFILE_OUT"

echo -e "----------------------------------------"
echo -e "${CYAN}✅ Tests completed.${NC}"

actual_coverage=$(tail -n 1 "$COVERAGE_OUT" | awk '{print $NF}' | sed 's/%//')
coverage_satisfied=$(echo "$actual_coverage >= $MIN_COVERAGE_PERCENT" | bc)

echo -e "${CYAN}📊 Code coverage: ${actual_coverage}%${NC}"

if [ "$coverage_satisfied" -eq 0 ]; then
    echo -e "${RED}❌ Minimum required coverage (${MIN_COVERAGE_PERCENT}%) not met.${NC}"
    exit 1
else
    echo -e "${GREEN}✔ Minimum coverage requirement satisfied.${NC}"
fi
