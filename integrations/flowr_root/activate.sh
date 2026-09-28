#!/usr/bin/env bash
# Source this file from a Bash shell: source integrations/flowr_root/activate.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Source this file so activation applies to the current shell.\n' >&2
    exit 2
fi

_molsteer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
_flowr_root="${_molsteer_root}/flowr_root"
case "${MOLSTEER_FLOWR_ENV:-overlay}" in
    overlay) _flowr_venv="${_flowr_root}/.venv-molsteer" ;;
    base) _flowr_venv="${_flowr_root}/.venv" ;;
    *) printf 'MOLSTEER_FLOWR_ENV must be overlay or base.\n' >&2; return 2 ;;
esac
if [[ ! -f "${_flowr_venv}/bin/activate" ]]; then
    printf 'FLOWR environment missing: %s\n' "${_flowr_venv}" >&2
    return 2
fi
source "${_flowr_venv}/bin/activate"
export MOLSTEER_ROOT="${_molsteer_root}"
export FLOWR_ROOT="${_flowr_root}"
export PYTHONPATH="${FLOWR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
unset _molsteer_root _flowr_root _flowr_venv
