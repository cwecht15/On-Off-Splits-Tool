# Testing

## Test Files

- `tests/test_data_correctness.py`
  - validates core math and filtering behavior
  - checks:
    - success-rate rule logic
    - on/off partition integrity
    - weighted EPA consistency
    - pass/run-only filter integrity

- `tests/test_golden_regression.py`
  - offense golden regression checks
  - fixed scenario:
    - team `ARZ`
    - player `00-0033553`
    - role `offense`
    - season `2021`
    - weeks `1-18`
  - validates split metrics, trend checkpoints, personnel checkpoints

- `tests/test_golden_regression_defense.py`
  - defense golden regression checks
  - fixed scenario:
    - team `BUF`
    - player `00-0036914`
    - role `defense`
    - season `2021`
    - weeks `1-18`
  - validates split metrics, trend checkpoints, personnel checkpoints

## Run Tests

Run full suite:

```powershell
python -m unittest tests\test_data_correctness.py tests\test_golden_regression.py tests\test_golden_regression_defense.py -v
```

Run only one file:

```powershell
python -m unittest tests\test_data_correctness.py -v
```

## Release Checklist

Before pushing app changes:
- run full test suite
- verify app starts locally
- verify one offense and one defense selection manually in UI
- push and confirm cloud redeploy health

## Updating Golden Tests

Golden tests intentionally fail if output changes.

If you intentionally changed business logic:
1. Recompute expected values from current logic.
2. Update constants in golden test files.
3. Re-run suite and confirm all pass.
