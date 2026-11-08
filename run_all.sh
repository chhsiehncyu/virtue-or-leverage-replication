#!/bin/bash
# Runs every step of the replication package in order and writes one log per script to the folder logs.
set -u
cd "$(dirname "$0")"
mkdir -p logs
for s in model.py endog_r.py quartic.py feasibility_bounds.py proof_concavity.py proof_concavity_L.py \
         proof_monotone_commission.py proof_welfare_rigid.py grid_evaluation.py profit_gain_flex.py \
         observations.py baseline_numbers.py check_reviewers.py rigid_tables.py tables_4_5.py nonuniform2.py \
         extensions_check.py continuous_quality_flex.py figures.py; do
  echo "running $s"
  if ! python3 "$s" > "logs/${s%.py}.log" 2>&1; then
    echo "$s failed; see logs/${s%.py}.log"
    exit 1
  fi
done
echo "all steps finished"
