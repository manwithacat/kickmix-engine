#!/bin/bash
# Re-run gates 2 and 3 into a new evidence directory; call when the machine is otherwise idle.
# Usage: rerun_when_idle.sh TAG   (writes outputs/engine-m1-2026-10-27/{single_core,scaling,kickmix_scaling}_TAG.json)
set -e
R=/Volumes/SSD/Kaggle/tools/kickmix-engine
E=/Volumes/SSD/Kaggle/competitions/google-verifiable-quantum-circuit-compilation/outputs/engine-m1-2026-10-27
K=/private/tmp/claude-501/-Volumes-SSD-Kaggle/bede4ed6-52b3-426f-9474-6b45a869e799/scratchpad/cxpop313/bin/python
T=$1; [ -e "$E/scaling_$T.json" ] && { echo "refusing to overwrite $E/scaling_$T.json"; exit 1; }
uptime
PYTHONPATH=$R/python $K $R/bench/single_core.py $E/single_core_$T.json --rounds 10
PYTHONPATH=$R/python $K $R/bench/scaling.py $E/scaling_$T.json --chunks 320 --rounds 5
PYTHONPATH=$R/python $K $R/bench/kickmix_scaling.py $E/kickmix_scaling_$T.json
