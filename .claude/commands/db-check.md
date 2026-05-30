Run a health check across all pipeline databases.

Run from /Users/shivam/Desktop/Programming/Quant:

```
/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 scripts/db_check.py
```

After the output, flag anything that looks wrong: ERR lines, sync mismatches where a DB is more than one snapshot behind, or FY2025_income below 400.
