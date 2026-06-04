#!/usr/bin/env bash
set -euo pipefail
# set -x

THIS_FILE=$(readlink -f "${BASH_SOURCE[0]}")
THIS_DIR=$(dirname "$THIS_FILE")
ROOT_DIR=$(dirname "$THIS_DIR")
WORKSPACE_DIR="$(dirname "$ROOT_DIR")"

. "$THIS_DIR/kash/kash.sh"
. "$THIS_DIR/kash_python.sh"

## Parse options
##

CI_STEP_NAME="Run tests"
RUN_SONAR=false
while getopts "sr:" option; do
    case $option in
        s) # enable SonarQube analysis and publish code quality & coverage results
            RUN_SONAR=true
            ;;
        r) # report outcome to slack
            CI_STEP_NAME=$OPTARG
            load_env_files "$WORKSPACE_DIR/development/common/SLACK_WEBHOOK_SERVICES.enc.env"
            trap 'slack_ci_report "$ROOT_DIR" "$CI_STEP_NAME" "$?" "$SLACK_WEBHOOK_SERVICES"' EXIT
            ;;
        *)
            ;;
    esac
done


## Init workspace
##

. "$WORKSPACE_DIR/development/workspaces/services/services.sh" kazarr

## Setup micromamba env
##

setup_micromamba_env "kazarr-test" "$THIS_DIR/test-environment.yml" true

## Run tests
##

run_python_lib_tests "$ROOT_DIR" "$RUN_SONAR"