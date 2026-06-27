#!/usr/bin/env bash
set -e
echo "Starting local LLM inference server..."
python -m agentic_security.llm.serve
