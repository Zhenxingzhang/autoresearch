#!/bin/bash
set -euo pipefail

export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
export CLAUDE_TLS_CERT_PATH=/turibolt_k8s_mounts/narrative/turi/cert.pem
export CLAUDE_TLS_KEY_PATH=/turibolt_k8s_mounts/narrative/turi/private.pem

apt-get update
apt-get install jq -y
curl https://pages.github.pie.apple.com/crypto-services/trust-apple-corp-root-cas/trust_apple_corp_root_cas.sh | bash
curl https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
\. "$HOME/.nvm/nvm.sh"
nvm install 22
nvm use 22
npm install -g @apple/claude-code --registry https://npm.apple.com

ln -s /miniforge/include/python3.10 /usr/include/python3.10