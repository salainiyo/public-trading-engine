# Security Policy

Never open an issue or pull request containing real credentials, broker
account details, access tokens, private keys or account emails.

## Reporting a vulnerability

Please report security issues privately through GitHub's "Report a
vulnerability" feature on the repository's Security tab. Include the affected
file, the impact and a minimal reproduction. You should receive an
acknowledgement within a few days.

## Credential handling

- All secrets are read from the local `.env` file, which is git-ignored.
- `.env.example` contains names and placeholders only.
- Broker credentials are encrypted at rest via `config/credentials.py`.
- Rotate any credential that may have been exposed, including SMTP app
  passwords and broker tokens.
