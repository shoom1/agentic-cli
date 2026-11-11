#!/usr/bin/env bash
# Which of CI's four part jobs a change runs: one "<job>=true|false" line each
# for kb, wheel, latex and docker (the workflow appends them to $GITHUB_OUTPUT).
#
# Usage: ci-jobs.sh [<base> <head>]
#
# Given a pull request's base and head commits, a job runs when the pull
# request touches what that job tests, counting only the pull request's own
# changes (the diff from where it left its base). With no commits (a push to
# develop or main), or a diff that cannot be read, every job runs. The offline
# suite is not decided here: it runs on every change.
set -u

# What every job is built from: a change here runs them all.
EVERY='^(\.github/workflows/ci\.yml|\.github/scripts/ci-jobs\.sh|pyproject\.toml|environment\.yml)$'

# The kb extra (sentence-transformers, trafilatura) changes what the memory
# package, web_fetch's page conversion and the knowledge tools do, and the
# logging setup quiets its loggers.
KB='^(src/agentic_cli/memory/|src/agentic_cli/tools/webfetch/|src/agentic_cli/tools/(knowledge_tools|_knowledge_base|arxiv_tools|pdf_utils)\.py$|src/agentic_cli/workflow/memory_services\.py$|src/agentic_cli/logging\.py$|tests/memory/|tests/tools/test_(kb_|webfetch|async_tools_never_block))'

# The built wheel: what it packages (examples/research_demo) and the tests
# that build it. A module added or removed under src/, and a file there that
# is not Python (package data), are checked below.
WHEEL='^(examples/research_demo/|tests/examples/(console_smoke\.py|test_[a-z_]*_wheel\.py)$)'

# compile_document, the path resolution it judges by, and its engine tests.
LATEX='^(src/agentic_cli/tools/document/|src/agentic_cli/paths\.py$|tests/tools/test_document_compile)'

# The sandbox, the descriptor-safe copies it stages through, the settings
# that name its image (CI pre-pulls that tag), and its live tests.
DOCKER='^(src/agentic_cli/tools/sandbox/|src/agentic_cli/file_utils\.py$|src/agentic_cli/workflow/settings\.py$|tests/tools/test_sandbox_docker)'

every() {
  for job in kb wheel latex docker; do echo "$job=true"; done
}

if [ $# -lt 2 ]; then
  every
  exit 0
fi
# --no-renames: a moved file is a removal and an addition, which the wheel
# check below counts.
if ! changes=$(git diff --name-status --no-renames "$1...$2" 2>/dev/null); then
  echo "ci-jobs: cannot read the diff $1...$2; running every job" >&2
  every
  exit 0
fi
paths=$(printf '%s\n' "$changes" | cut -f2)

touches() {
  printf '%s\n' "$paths" | grep -qE "$1"
}

packaging() {
  touches "$WHEEL" && return 0
  printf '%s\n' "$paths" | grep -E '^src/' | grep -qvE '\.py$' && return 0
  printf '%s\n' "$changes" | grep -qE '^[AD][[:space:]]+src/'
}

flag() {
  if "$@"; then echo true; else echo false; fi
}

if touches "$EVERY"; then
  every
  exit 0
fi
echo "kb=$(flag touches "$KB")"
echo "wheel=$(flag packaging)"
echo "latex=$(flag touches "$LATEX")"
echo "docker=$(flag touches "$DOCKER")"
