# Security

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/StartLuxLabs/StartLux-Decision/security/advisories/new) and not in
a public issue. Include the version or commit, the command line of the server and the smallest request that shows
the problem. We will acknowledge the report and keep you informed while we work on a fix.

The server is built to face the internet only with API keys, TLS and request limits switched on; see
[docs/serving_publicly.md](docs/serving_publicly.md). The vLLM servers it launches listen on 127.0.0.1 and have no
authentication of their own, so never expose their ports.
