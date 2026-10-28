#!/bin/bash
# Gates 3 (memory) and 4 (speed): full DQI verification, the official CLI on the kickmix fork vs the engine CLI, at 1 and
# 10 workers/threads, alternating order, ROUNDS rounds, every run under bench/memwatch.py (process-tree peak footprint).
# The engine is run cold (empty program cache, compile included) and warm (cached program) separately.
#
# Usage: dqi_gate34.sh OUTDIR TAG [ROUNDS] [CIRCUIT]
# Environment (each side can point at an installed wheel's interpreter):
#   FORK_PY          interpreter whose `kickmix` is the fork build used for the official verifier (required)
#   ENGINE_PY        interpreter with kickmix_engine importable and kickmix importable (the verifier imports it);
#                    default: FORK_PY with the engine source tree on PYTHONPATH
#   ENGINE_SRC       1 = put this repo's python/ on PYTHONPATH for the engine (default 1 if ENGINE_PY is unset, else 0)
#   VERIFIER         official verify_submission_dqi.py (required)
#   CACHE_ROOT       directory for the engine's program caches (default: OUTDIR/../<TAG>-cache, removed at the end)
#   VARIANTS         engine variants to run, default "cold warm"
#   WORKERS          parallelism levels, default "1 10"
# Output files: gate34_TAG_fork_wN_rR.json, gate34_TAG_engine-cold_tN_rR.json, gate34_TAG_engine-warm_tN_rR.json
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
O=$1; T=$2; N=${3:-3}; C=$4
: "${FORK_PY:?set FORK_PY}"; : "${VERIFIER:?set VERIFIER}"; : "${C:?circuit path required}"
if [ -z "$ENGINE_PY" ]; then ENGINE_PY=$FORK_PY; ENGINE_SRC=${ENGINE_SRC:-1}; else ENGINE_SRC=${ENGINE_SRC:-0}; fi
VARIANTS=${VARIANTS:-"cold warm"}; WORKERS=${WORKERS:-"1 10"}
CACHE_ROOT=${CACHE_ROOT:-$(dirname "$O")/$T-cache}
EPATH=""; [ "$ENGINE_SRC" = 1 ] && EPATH="$R/python"
export KICKMIX_ENGINE_DQI_VERIFIER=$VERIFIER
mkdir -p "$O" "$CACHE_ROOT"
{
  echo "date: $(date)"; uptime
  echo "FORK_PY=$FORK_PY"; $FORK_PY -c 'import kickmix,sys,hashlib;print("fork kickmix", kickmix.__file__, hashlib.sha256(open(kickmix.__file__,"rb").read()).hexdigest(), sys.version.split()[0])'
  echo "ENGINE_PY=$ENGINE_PY ENGINE_SRC=$ENGINE_SRC"
  PYTHONPATH=$EPATH $ENGINE_PY -c 'import kickmix_engine as k,kickmix,hashlib;print("engine", k.__file__, k.__version__, hashlib.sha256(open(k._engine.__file__,"rb").read()).hexdigest(), "verifier kickmix", kickmix.__file__)'
  echo "VERIFIER=$VERIFIER $(shasum -a 256 "$VERIFIER" | cut -c1-64)"; echo "CIRCUIT=$C"; echo "VARIANTS=$VARIANTS WORKERS=$WORKERS ROUNDS=$N"
} | tee "$O/gate34_${T}_environment.txt"
engine_run() {  # variant workers outfile
  local cache="$CACHE_ROOT/$1"
  if [ "$1" = cold ]; then rm -rf "$cache"; fi
  if [ "$1" = warm ] && [ ! -d "$cache/programs" ]; then  # untimed warm-up, once
    PYTHONPATH=$EPATH $ENGINE_PY -m kickmix_engine verify dqi "$C" --threads "$2" --cache-dir "$cache" > /dev/null
  fi
  PYTHONPATH=$EPATH $FORK_PY "$R/bench/memwatch.py" "$3" -- env PYTHONPATH=$EPATH $ENGINE_PY -m kickmix_engine verify dqi "$C" --threads "$2" --cache-dir "$cache" --timing
}
for r in $(seq 1 "$N"); do
  for w in $WORKERS; do
    whos="fork"; for v in $VARIANTS; do whos="$whos engine-$v"; done
    [ $((r % 2)) = 0 ] && whos=$(echo $whos | awk '{for (i = NF; i > 0; i--) printf "%s ", $i}')
    for who in $whos; do
      if [ "$who" = fork ]; then f=$O/gate34_${T}_fork_w${w}_r${r}.json; else f=$O/gate34_${T}_${who}_t${w}_r${r}.json; fi
      [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
      echo "round $r $who n=$w $(uptime)"
      if [ "$who" = fork ]; then
        $FORK_PY "$R/bench/memwatch.py" "$f" -- $FORK_PY "$VERIFIER" "$C" --max-workers "$w"
      else
        engine_run "${who#engine-}" "$w" "$f"
      fi
    done
  done
done
uptime
[ -z "$KEEP_CACHE" ] && rm -rf "$CACHE_ROOT"
exit 0
