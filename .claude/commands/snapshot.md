Run a full pipeline snapshot for the given date: factors → models → risk → barra.

Date argument: $ARGUMENTS

Steps (run each sequentially, wait for each to finish before starting the next):
1. `/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 create_factors.py --date <date>`
2. `/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 create_models.py --date <date>`
3. `/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 create_risk.py --date <date>`
4. `/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 create_barra.py --date <date>`

All commands must be run from /Users/shivam/Desktop/Programming/Quant.

If no date is given, use today's date (YYYY-MM-DD format).

After each step, print a one-line status: the script name, whether it succeeded or failed, and any key output number (e.g. "factors: ✓  19,111 rows" or "models: ✗  error: ..."). If a step fails, stop and show the error — do not continue to the next step.

When all four complete, print a summary table:
  Script     | Status | Key metric
  -----------|--------|------------
  factors    | ✓      | N rows
  models     | ✓      | N rows
  risk       | ✓      | N stocks
  barra      | ✓      | snapshot_date
