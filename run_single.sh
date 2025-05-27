#!/bin/bash

# Verifica che un argomento sia stato passato
if [ -z "$1" ]; then
  echo "Uso: $0 <numero_step>"
  exit 1
fi

N_STEPS=$1

# Avvia una sessione tmux chiamata "train_n"
SESSION="train_$N_STEPS"
tmux new-session -d -s "$SESSION"

# Esegui i comandi all'interno della sessione tmux
tmux send-keys -t "$SESSION" "conda activate pinn" C-m
tmux send-keys -t "$SESSION" "python main.py --steps $N_STEPS && tmux kill-session -t $SESSION" C-m

echo "Sessione tmux '$SESSION' avviata con $N_STEPS step. Si chiuderà automaticamente al termine."
