# Contributing

Contributions are welcome. This project is a focused, single-strategy bot, so
please keep changes small and aligned with that scope.

## Getting started

1. Fork the repository and create a branch from `main`.
2. Use Python 3.12 and install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` for local configuration; never commit `.env`.

## Before opening a pull request

- Run `python3 -m pytest tests/unit/ tests/integration/ -v`.
- Keep strategy and rulebook changes covered by tests.
- Do not add deployment automation, tracking code or real credentials.
- Update the README and CHANGELOG when behavior changes.

Pull requests are checked by CI. Describe what changed and why.
