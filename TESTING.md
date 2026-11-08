# Testing

Tests never read `.env` and never touch the camera, the Claude API or the network. Three tiers, each a folder under
`tests/`; the folder sets the pytest marker (`tests/conftest.py`), so a test is never tagged by hand.

| Tier | Folder | Rule |
|---|---|---|
| unit | `tests/unit/` | Pure logic. No files, database, sleeps, subprocesses, threads or network. Under 50 ms each. Mirrors the source tree: `tests/unit/test_<m>.py` for `birdbrain/<m>.py`, `tests/unit/web/` for `birdbrain/web/`, `tests/unit/scripts/` for `scripts/`. |
| component | `tests/component/` | One real local dependency: a temp database, temp files, a FastAPI TestClient, a fake tool on PATH. |
| integration | `tests/integration/` | Real services, devices or network, repo-wide checks, model files. May skip when a file is missing. |

`tests/unit/conftest.py` enforces the unit rule: inside a unit test, `open` for writing, `sqlite3.connect`,
`time.sleep`, `subprocess`, `Thread.start`, `tmp_path` and `capfd` fail with "unit tests are pure; move this test to
tests/component". The data paths point at a folder that does not exist.

Two extra markers: `smoke` (`@pytest.mark.smoke`, the short set the deploy runs on atmos and the mirror with
`pytest -m smoke`; never the full suite there, the old 2 GB mirror host ran out of memory) and `guard`
(`tests/integration/test_no_personal_info.py`: no owner name, account, repo URL, email or home IP in tracked files).

## Where a new test goes

- Can it run with no file, database, clock or process? `tests/unit/`, in the file that mirrors the module.
- Does it need a temp database, temp files or the web app? `tests/component/`.
- Does it touch a device, an API, the network or the whole repo? `tests/integration/`.
- Check for existing coverage first (`grep -rn "def test_" tests/ | grep <topic>`); extend a test rather than add a duplicate.

Mock at the boundary (the camera, the API client, the HTTP layer), not inside the code under test. Use the frozen
clock (`FROZEN_NOW` in `tests/helpers/clock.py`, applied by the base conftest); never a real sleep.

## Commands

```bash
uv run pytest tests/unit                    # fast feedback, about 3 s
uv run pytest tests/unit/test_db.py         # one file
uv run pytest tests/component -n 4          # the component tier, in parallel
uv run pytest                               # everything, before landing
uv run pytest -n 4                          # everything, in parallel (pytest-xdist is a dev dependency)
uv run --with pytest-randomly pytest -p randomly --randomly-seed=1   # random order
uv run pytest tests/unit -p no:cacheprovider --junitxml=/tmp/unit.xml \
  && uv run python scripts/check_test_budget.py /tmp/unit.xml       # the unit tier's runtime budget
uv run python scripts/select_tests.py main   # component tests for the paths changed since main
```

Budget: the unit tier must sum to under 30 s and no unit test may take over 0.25 s (over 50 ms is a warning).
`TEST_BUDGET_TOTAL` and `TEST_BUDGET_TEST` override the limits on a slow machine.

## CI (`.github/workflows/tests.yml`)

- Every push and pull request: the unit tier, its budget check, the guard tests, then the component tests for the changed paths.
- Pull requests and pushes to main: the whole component tier with `-n 4`.
- Nightly and on demand: the whole suite in file order, then again in random order (seed = run number, printed).
