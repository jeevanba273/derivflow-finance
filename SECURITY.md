# Security Policy

## Supported Versions

Security fixes are provided for the latest release series. Please ensure you are
running a current version before reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in DERIVFLOW-FINANCE, please report it
**privately**. Do **not** open a public GitHub issue, pull request, or
discussion for security vulnerabilities, as that could expose users to risk
before a fix is available.

Instead, report it privately via email to **jeevanba273@gmail.com**. Please
include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a proof-of-concept where possible.
- The affected version(s) and your environment details.

### Response expectations

- We aim to acknowledge your report within **3 business days**.
- We will provide an initial assessment and remediation plan within
  **10 business days**.
- Once a fix is released, we will credit the reporter unless anonymity is
  requested.

## Scope and a note on data sources

DERIVFLOW-FINANCE can pull **live market data over the network via
[`yfinance`](https://pypi.org/project/yfinance/)** (Yahoo Finance). That data is
provided by a third party and is outside the maintainers' control. Treat any
externally fetched data as untrusted input, validate it before use in
production, and be aware that network access introduces availability and
data-integrity considerations independent of this library's code.
