# Security Policy

## Supported Versions

Security fixes are provided for the latest stable release of SysIdentPy. Older
releases are not actively supported with security updates.

| Version | Supported |
| ------- | --------- |
| Latest release | Yes |
| Older releases | No |

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately through GitHub
Security Advisories:

<https://github.com/wilsonrljr/sysidentpy/security/advisories/new>

Please include as much detail as possible so the issue can be triaged
effectively:

- the affected SysIdentPy version;
- the operating system, Python version, and relevant dependency versions;
- a minimal reproducible example or clear reproduction steps;
- the expected security impact;
- whether the vulnerability is already public or known to third parties.

Reports will be reviewed privately as soon as reasonably possible. If the report
is accepted, a fix will be prepared privately and released with a coordinated
GitHub advisory when appropriate. If the report is not considered a security
vulnerability, it may be closed with an explanation and, when useful, redirected
to the regular issue tracker.

## Security Scope

Issues that may qualify as security vulnerabilities include:

- unintended code execution, file access, or data corruption caused by
  apparently benign inputs;
- vulnerabilities in `save_model`, `load_model`, or serialization workflows when
  used with files that the caller reasonably trusts;
- vulnerabilities in vendored dependencies that are exploitable through
  SysIdentPy.

The following are generally not considered security vulnerabilities:

- numerical differences, statistical instability, convergence behavior, or model
  performance differences;
- high CPU, memory, or runtime cost from model training, structure selection,
  examples, notebooks, or large datasets;
- code execution caused by loading untrusted `.syspy` or pickle files, because
  pickle-based files must be treated as executable code;
- unsafe use of scripts, notebooks, models, or datasets obtained from untrusted
  sources.

## Safe Usage Notes

Install SysIdentPy and its dependencies from trusted sources, keep them updated,
and avoid running examples or workflows with untrusted data unless they are
isolated from sensitive files, credentials, and networks.

Do not load `.syspy` model files from unknown or untrusted sources. SysIdentPy
model files use Python pickle serialization, and loading a pickle file can
execute arbitrary code.
