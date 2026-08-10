#!/usr/bin/env bash
set -euo pipefail
# set -x

THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")
ROOT_DIR=$(dirname "$THIS_DIR")
WORKSPACE_DIR="$(dirname "$ROOT_DIR")"

. "$THIS_DIR/kash/kash.sh"

## Parse options
##

CI_STEP_NAME="Run tests"
PUBLISH=false
while getopts "p" option; do
    case $option in
        p) # publish packages to the configured repository
            PUBLISH=true
            ;;
        *)
            ;;
    esac
done


## Init workspace
##

. "$WORKSPACE_DIR/development/workspaces/services/services.sh" kazarr

## Load env files
##

load_env_files "$WORKSPACE_DIR/development/common/kalisio_pypi.enc.env"

## Setup micromamba env
##

setup_python_env

## Build and publish
##

export UV_PUBLISH_TOKEN=$(decrypt_stdout "$WORKSPACE_DIR/development/common/KALISIO_PYPI_TOKEN.enc.value")

build_and_publish_python_lib "$ROOT_DIR/conversion_tool" "$PUBLISH" "$PUBLISH_URL"

unset UV_PUBLISH_TOKEN