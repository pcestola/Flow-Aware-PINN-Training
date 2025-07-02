#!/bin/bash

# Verifica che vengano passati i parametri necessari
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Uso: $0 <steps> <numero_ripetizioni>"
  exit 1
fi

STEPS=$1
N_REPEAT=$2

AVAILABLE_GPUS=(0 1 3 4 5 6 7)

SESSION="pinnacle"
tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "conda activate pinnacle" C-m
tmux send-keys -t "$SESSION" "python benchmark.py --steps $STEPSZ --repeat $N_REPEAT && tmux kill-session -t $SESSION" C-m
echo "✅ Avviata sessione: steps=$STEP, repeat=$N_REPEAT"
