#!/bin/bash

cover_pkgs=$(go list ./... | grep -v "$EXCLUDED_COVERAGE_PATH" | tr '\n' ',')
coverage_out=coverage
profile_out=profile

go test -coverpkg=$cover_pkgs -coverprofile=$profile_out ./... || exit 1
go tool cover -func $profile_out >$coverage_out

actual_coverage=$(cat $coverage_out | tail -n 1 | sed 's/.*[[:blank:]]//' | sed 's/%//')
minimum_coverage=$MIN_COVERAGE_PERCENT
echo "Code coverage: $actual_coverage%"

coverage_satisfied=$(echo "$actual_coverage >= $minimum_coverage" | bc)
if [ "$coverage_satisfied" -eq 0 ]; then
    echo "Minimum coverage $minimum_coverage% was not satisfied."
    exit 1
fi
