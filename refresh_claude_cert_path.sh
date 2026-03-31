#!/bin/bash

export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
export CLAUDE_TLS_CERT_PATH=/turibolt_k8s_mounts/narrative/turi/cert.pem
export CLAUDE_TLS_KEY_PATH=/turibolt_k8s_mounts/narrative/turi/private.pem