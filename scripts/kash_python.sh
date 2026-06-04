MICROMAMBA_VERSION="latest"

install_micromamba() {
    export PREFIX_LOCATION="${PREFIX_LOCATION:-${HOME}/micromamba}"
    # Mandatory for micromamba to work post installation
    export MAMBA_ROOT_PREFIX="$PREFIX_LOCATION"
    export BIN_FOLDER="${BIN_FOLDER:-${HOME}/.local/bin}"
    export INIT_YES="no" 
    export CONDA_FORGE_YES="yes"

    curl -Lso /tmp/install_micromamba.sh https://micro.mamba.pm/install.sh
    bash /tmp/install_micromamba.sh < /dev/null

    export MAMBA_EXE="${BIN_FOLDER}/micromamba"
    export PATH="$(dirname "$MAMBA_EXE"):$PATH"
    eval "$("$MAMBA_EXE" shell hook -s bash)"
}

setup_micromamba_env() {
    # Ensure root prefix is set for micromamba, even if CI has changed its shell
    export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"

    local env_name="$1"
    local env_file="${2:-environment.yml}" 
    local auto_activate="${3:-false}"

    if [ -z "$env_name" ]; then
        echo "Error: Environment name is required."
        return 1
    fi

    if [ ! -f "$env_file" ]; then
        echo "Error: The file '$env_file' is not found."
        return 1
    fi

    echo "Creating environment '$env_name' from '$env_file'..."
    micromamba create -f "$env_file" -n "$env_name" -y

    if [ $? -eq 0 ]; then
        echo "Environment '$env_name' created and configured successfully !"
        
        if [ "$auto_activate" = "true" ] || [ "$auto_activate" = "1" ]; then
        echo "Automatic activation of the environment '$env_name'..."
        eval "$(micromamba shell hook --shell bash)"
        micromamba activate "$env_name"
        fi
    else
        echo "Error occurred while creating the environment."
        return 1
    fi
}

run_pytest() {
    local final_status=0

    echo "Running ruff checks..."
    if ! ruff check; then
        final_status=1
    fi

    echo "Running pytest with coverage..."
    if ! pytest --cov --cov-report=xml:coverage.xml "$@"; then
        final_status=1
    fi

    return $final_status
}

run_python_lib_tests() {
    local ROOT_DIR="$1"
    local CODE_COVERAGE="${2:-false}"
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

    ## Publish code coverage
    ##

    if [ "$CODE_COVERAGE" = true ]; then
        send_coverage_to_cc "$CC_TEST_REPORTER_ID"
    fi
}