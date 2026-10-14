"""A throwaway self-signed certificate for TLS tests, made with the openssl command (None when it is missing)."""
from __future__ import annotations

import os
import shutil
import subprocess


def self_signed(directory: str) -> tuple[str, str] | None:
    """(certificate file, key file) for CN=localhost, valid one day; None when openssl is not available."""
    openssl = shutil.which("openssl")
    if openssl is None:
        return None
    cert, key = os.path.join(directory, "cert.pem"), os.path.join(directory, "key.pem")
    done = subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                           "-days", "1", "-subj", "/CN=localhost"], capture_output=True)
    return (cert, key) if done.returncode == 0 and os.path.isfile(cert) else None
