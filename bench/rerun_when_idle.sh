#!/bin/bash
# Re-run gates 2 and 3 into a new evidence directory; call when the machine is otherwise idle.
# Usage: FORK_PY=/path/to/python-with-kickmix EVIDENCE=DIR KMB=invsqrt.kmb KICKMIX_ENGINE_VERIFIERS=DIR rerun_when_idle.sh TAG
# (writes DIR/{single_core,scaling,kickmix_scaling}_TAG.json; runs the engine from this source tree, built by ./build.sh)
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
E=${EVIDENCE:?set EVIDENCE}
K=${FORK_PY:?set FORK_PY}
KMB=${KMB:?set KMB to the invsqrt .kmb}
T=$1; [ -e "$E/scaling_$T.json" ] && { echo "refusing to overwrite $E/scaling_$T.json"; exit 1; }
uptime
PYTHONPATH=$R/python $K $R/bench/single_core.py $E/single_core_$T.json --kmb "$KMB" --rounds 10
PYTHONPATH=$R/python $K $R/bench/scaling.py $E/scaling_$T.json --kmb "$KMB" --chunks 320 --rounds 5
PYTHONPATH=$R/python $K $R/bench/kickmix_scaling.py $E/kickmix_scaling_$T.json --kmb "$KMB"
