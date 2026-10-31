#!/usr/bin/env bash
# Source this in the lab root once per session. No command is executed for you.
mkdir -p runs
export LAB_RUN
LAB_RUN="$(mktemp -d "$PWD/runs/reader-XXXXXXXX")" || return
printf 'LAB_RUN=%s\n' "$LAB_RUN"
