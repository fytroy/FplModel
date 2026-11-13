# Part 4 implementation plan: deadline alerts

Spec: `docs/superpowers/specs/2026-10-12-deadline-alerts-design.md`.

1. [x] `fpl/alerts.py` (pure): `next_deadline`, `stage_for`, `decisions`, `changed`, `late_action`, `short_message`. Tests in `tests/test_alerts.py`.
2. [x] `fpl/api.py`: `send_ntfy` (JSON POST to ntfy.sh, retried). `fpl/storage.py`: `save_alert`/`load_alert` (`data/alerts/gw<N>.json`). Window bounds in `fpl/settings.py`.
3. [x] `recommend.py`: `run(team_id, horizon, log)` returns the `Advice`; `main()` prints it as before.
4. [x] `alert.py`: `gate --mode` and `run --stage [--dry-run|--test]`. Early is sent once, and late only if the decisions changed or early was missed. Both save the record (except dry runs and tests). Flow tests use a faked pipeline and ntfy.
5. [x] `.github/workflows/deadline-alerts.yml`: hourly gate → advise. Cache keys mark what has been sent. Past seasons are cached. Failure pings go out once per gameweek and stage, or once a day for the gate.
6. [x] CLAUDE.md: command, workflow and setup steps.
7. [ ] After merge: manual `check`, `dry-run` and `test-send` runs on GitHub. Record the timings and solver status, then check minutes usage after a week.
