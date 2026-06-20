#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

output="$("$repo_root/ai_pipeline.sh" --cleanup-test 2>&1)"
printf '%s\n' "$output"

if [[ "$output" != *"cleanup test passed"* ]]; then
    echo "expected cleanup test success message" >&2
    exit 1
fi

signal_output="$(mktemp)"
temp_record="$(mktemp)"
child_record="$(mktemp)"
ready_file="$(mktemp)"
rm -f "$ready_file"

cleanup_test_files() {
    rm -f "$signal_output" "$temp_record" "$child_record" "$ready_file"
}
trap cleanup_test_files EXIT

AI_PIPELINE_CLEANUP_TEST_TEMP_RECORD="$temp_record" \
AI_PIPELINE_CLEANUP_TEST_CHILD_RECORD="$child_record" \
AI_PIPELINE_CLEANUP_SIGNAL_READY="$ready_file" \
    "$repo_root/ai_pipeline.sh" --cleanup-signal-test >"$signal_output" 2>&1 &
pipeline_pid=$!

for _ in {1..50}; do
    if [[ -f "$ready_file" && -s "$temp_record" && -s "$child_record" ]]; then
        break
    fi
    sleep 0.1
done

if [[ ! -f "$ready_file" || ! -s "$temp_record" || ! -s "$child_record" ]]; then
    cat "$signal_output"
    echo "cleanup signal test failed to start" >&2
    kill "$pipeline_pid" 2>/dev/null || true
    wait "$pipeline_pid" 2>/dev/null || true
    exit 1
fi

temp_dir="$(cat "$temp_record")"
child_pid="$(cat "$child_record")"

if [[ ! -d "$temp_dir" ]]; then
    cat "$signal_output"
    echo "cleanup signal test did not create temp workspace: $temp_dir" >&2
    kill "$pipeline_pid" 2>/dev/null || true
    wait "$pipeline_pid" 2>/dev/null || true
    exit 1
fi

kill -TERM "$pipeline_pid"
set +e
wait "$pipeline_pid"
status=$?
set -e

cat "$signal_output"

if [[ "$status" -ne 143 ]]; then
    echo "cleanup signal test expected exit status 143, got $status" >&2
    exit 1
fi

if [[ -e "$temp_dir" ]]; then
    echo "cleanup signal test left temp workspace behind: $temp_dir" >&2
    exit 1
fi

if kill -0 "$child_pid" 2>/dev/null; then
    echo "cleanup signal test left child process running: $child_pid" >&2
    kill "$child_pid" 2>/dev/null || true
    exit 1
fi

echo "cleanup signal test passed: TERM removed temp workspace and stopped child process"
