#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

output="$("$repo_root/ai_pipeline.sh" --cleanup-test 2>&1)"
printf '%s\n' "$output"

if [[ "$output" != *"cleanup test passed"* ]]; then
    echo "expected cleanup test success message" >&2
    exit 1
fi
