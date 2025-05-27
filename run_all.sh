#!/bin/bash

# Verifica che venga passato il numero di ripetizioni
if [ -z "$1" ]; then
  echo "Uso: $0 <numero_ripetizioni>"
  exit 1
fi

N_REPEAT=$1
STEP_VALUES=(1 2 4 8 16 32 64)

# Lista GPU da usare, escludendo la 2
AVAILABLE_GPUS=(0 1 3 4 5 6 7)

# Verifica che ci siano abbastanza GPU
if [ "${#STEP_VALUES[@]}" -gt "${#AVAILABLE_GPUS[@]}" ]; then
  echo "Errore: servono almeno ${#STEP_VALUES[@]} GPU disponibili (escludendo la 2)"
  exit 1
fi

for IDX in "${!STEP_VALUES[@]}"; do
  STEP=${STEP_VALUES[$IDX]}
  GPU_ID=${AVAILABLE_GPUS[$IDX]}

  SESSION="train_${STEP}"

  tmux new-session -d -s "$SESSION"
  tmux send-keys -t "$SESSION" "conda activate pinn" C-m
  tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --steps $STEP --repeat $N_REPEAT && tmux kill-session -t $SESSION" C-m

  echo "✅ Avviata sessione '$SESSION': steps=$STEP, repeat=$N_REPEAT, GPU=$GPU_ID"
done
