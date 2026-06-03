MICROMAMBA_VERSION="latest"

run_pytest() {
    ruff check
    pytest "$@"
}

run_python_lib_tests() {
    local ROOT_DIR="$1"
    local WORKSPACE_DIR
    WORKSPACE_DIR="$(dirname "$ROOT_DIR")"

    local LIB
    LIB=$(get_toml_value "$ROOT_DIR/pyproject.toml" "project.name")
    local VERSION
    VERSION=$(get_toml_value "$ROOT_DIR/pyproject.toml" "project.version")

    ## Run tests
    ##

    echo "About to run tests for $LIB v$VERSION..."

    cd "$ROOT_DIR"
    export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    run_pytest
}