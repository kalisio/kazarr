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

DEFAULT_DEBIAN_VER=bookworm
DEBIAN_VER=$DEFAULT_DEBIAN_VER
PUBLISH=false
CI_STEP_NAME="Build service"
while getopts "pr:" option; do
    case $option in
        p) # publish app
            PUBLISH=true
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

init_lib_infos "$ROOT_DIR"

ensure_yq
NAME=$(yq -p toml '.project.name' "$ROOT_DIR/pyproject.toml")
VERSION=$(yq -p toml '.project.version' "$ROOT_DIR/pyproject.toml")
GIT_TAG=$(get_lib_tag)

# Strip @kalisio part
NAME=${NAME#*/}

echo "About to build $NAME v$VERSION ..."

load_env_files "$WORKSPACE_DIR/development/common/kalisio_dockerhub.enc.env"
load_value_files "$WORKSPACE_DIR/development/common/KALISIO_DOCKERHUB_PASSWORD.enc.value"

## Build container
##

IMAGE_NAME="$KALISIO_DOCKERHUB_URL/kalisio/$NAME"
IMAGE_SHORT_TAG=latest

if [[ -n "$GIT_TAG" ]]; then
    IMAGE_SHORT_TAG=$VERSION
fi

IMAGE_TAG="$IMAGE_SHORT_TAG"

begin_group "Building container $IMAGE_NAME:$IMAGE_TAG ..."

docker login --username "$KALISIO_DOCKERHUB_USERNAME" --password-stdin "$KALISIO_DOCKERHUB_URL" < "$KALISIO_DOCKERHUB_PASSWORD"
# DOCKER_BUILDKIT is here to be able to use Dockerfile specific dockerginore (app.Dockerfile.dockerignore)
DOCKER_BUILDKIT=1 docker build \
    -f Dockerfile \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    "$ROOT_DIR"

if [ "$PUBLISH" = true ]; then
    docker push "$IMAGE_NAME:$IMAGE_TAG"
fi

docker logout "$KALISIO_DOCKERHUB_URL"

end_group "Building container $IMAGE_NAME:$IMAGE_TAG ..."