#!/bin/bash
. /opt/venv/bin/activate
exec python -m agentic_security serve --host 0.0.0.0
