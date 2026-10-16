#!/usr/bin/env bash
#
# Remove generated artifacts.  Safe to re-run at any time.
#
# Tiered on purpose.  The default tier removes only scaffolding that a rebuild
# or a rerun reproduces exactly.  Everything that cost a cluster window, a
# simulator run, or a 900 MB generation pass is opt-in, because this repo has
# already lost measurement data once (see docs/recovering-shifted-benchmark-csv.md)
# and a cleaner is a bad place to find out that something was irreplaceable.
#
#   ./clear.sh              DRY RUN: print what would go, delete nothing
#   ./clear.sh -f           actually delete (build + run scaffolding, caches, venv)
#   ./clear.sh --sim        + simulator traces and stats
#   ./clear.sh --jobs       + appliance job metadata
#   ./clear.sh --logs       + root sweep/test logs
#   ./clear.sh --all        all of the above (still not matrices or results)
#   ./clear.sh --matrices   + the generated matrix cache (asks first)
#
# NEVER removed, at any tier except the explicit --matrices:
#   cluster/results/   measurement CSVs — the record behind every paper number
#   matrices/          generated matrix cache, ~900 MB, hours to rebuild
#   memory/ docs/ src/ bench/configs/ tests/ ai-docs/ .git/

set -euo pipefail

# ---------------------------------------------------------------------------
# Run from the repo root whatever the caller's cwd, and refuse to run anywhere
# that is not this repo.  Everything below is rm -rf; a wrong cwd is the one
# failure mode that matters.
# ---------------------------------------------------------------------------
cd "$(dirname "${BASH_SOURCE[0]}")"

for sentinel in config.py run.py src/csl/pe.csl cluster/worker_runtime.py; do
    if [[ ! -e $sentinel ]]; then
        echo "refusing to run: $PWD does not look like the spmv repo (no $sentinel)" >&2
        exit 1
    fi
done

PROTECTED=(cluster/results matrices memory docs src bench/configs tests ai-docs .git)

# Dry run unless -f.  Deliberate: every tier below is rm -rf, and the cost of
# an accidental bare invocation is unbounded while the cost of typing -f is one
# character.
DRY_RUN=1
WANT_SIM=0
WANT_JOBS=0
WANT_LOGS=0
WANT_MATRICES=0

usage() { sed -n '3,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run) DRY_RUN=1 ;;
        -f|--force)   DRY_RUN=0 ;;
        --sim)        WANT_SIM=1 ;;
        --jobs)       WANT_JOBS=1 ;;
        --logs)       WANT_LOGS=1 ;;
        --matrices)   WANT_MATRICES=1 ;;
        --all)        WANT_SIM=1; WANT_JOBS=1; WANT_LOGS=1 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Collection.  Targets are gathered first and deleted second, so a dry run and
# a real run walk exactly the same list.
# ---------------------------------------------------------------------------
TARGETS=()

is_protected() {
    local path=${1#./}
    # Derived caches are removable wherever they live, protected trees included:
    # they hold no measurement and no source, and tests/__pycache__ is as much
    # scaffolding as any other.  Checked before the guard list for that reason.
    case ${path##*/} in
        __pycache__|*.egg-info) return 1 ;;
    esac
    for guard in "${PROTECTED[@]}"; do
        [[ $path == "$guard" || $path == "$guard"/* ]] && return 0
    done
    return 1
}

add() {
    for path in "$@"; do
        [[ -e $path || -L $path ]] || continue
        if is_protected "$path"; then
            echo "BUG: refusing to delete protected path $path" >&2
            exit 1
        fi
        TARGETS+=("$path")
    done
}

# -- tier 1: build and run scaffolding (always) ------------------------------
# cluster_work/ and sweep_work/ are rebuilt by the next run; out/ and the
# artifact pointers are compiler output; File is a 0-byte stray.
add cluster_work sweep_work out .venv
add artifact.json artifact_path.json File
# wio_flows_tmpdir.* are simulator scratch dirs, not the traces themselves.
for d in wio_flows_tmpdir.*; do add "$d"; done
# The worker writes this basename; sdk/launcher.py downloads and unlinks it,
# so a copy in the repo root is a leftover from an interrupted run.
add benchmark_results.csv

# Python and tool caches.
add .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov build dist
while IFS= read -r -d '' d; do add "$d"; done < <(
    find . \( -path ./.git -o -path ./matrices -o -path ./cluster/results \) -prune -o \
           \( -type d \( -name __pycache__ -o -name '*.egg-info' \) \) -print0 2>/dev/null
)

# -- tier 2: simulator output (--sim) ----------------------------------------
# simfab_traces/ is ~12 MB of CTF and is how the cycle-6820 deadlock was
# diagnosed; it cannot be regenerated without re-running that exact config.
if (( WANT_SIM )); then
    add simfab_traces sim_stats.json simconfig.json sim.log
fi

# -- tier 3: appliance job metadata (--jobs) ---------------------------------
if (( WANT_JOBS )); then
    for f in wsjob-*-cluster-details.json; do add "$f"; done
    add run_meta.json
fi

# -- tier 4: root logs (--logs) ----------------------------------------------
# Sweep and test logs. Forensic value: these are the only record of runs whose
# CSV rows were lost to the positional-append corruption.
if (( WANT_LOGS )); then
    for f in ./*.log; do add "$f"; done
fi

# -- tier 5: matrix cache (--matrices), confirmed interactively --------------
if (( WANT_MATRICES )); then
    size=$(du -sh matrices 2>/dev/null | cut -f1 || echo '?')
    echo "matrices/ is $size and takes a long generation pass to rebuild."
    if (( DRY_RUN )); then
        echo "would ask before removing matrices/ (dry run)"
    else
        read -r -p "remove it? [y/N] " reply
        [[ ${reply:-N} == [yY] ]] && TARGETS+=(matrices) \
            || echo "keeping matrices/"
    fi
fi

# ---------------------------------------------------------------------------
# Report and delete.
# ---------------------------------------------------------------------------
if (( ${#TARGETS[@]} == 0 )); then
    echo "Nothing to clean."
    exit 0
fi

total=$(du -sk "${TARGETS[@]}" 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s/1024}')

for path in "${TARGETS[@]}"; do
    printf '  %s  %s\n' "$( (( DRY_RUN )) && echo 'would remove' || echo 'removing   ')" "$path"
done

if (( DRY_RUN )); then
    echo "Dry run: ${#TARGETS[@]} path(s), ${total} MB would be freed. Nothing deleted."
    echo "Re-run with -f to delete."
    exit 0
fi

rm -rf -- "${TARGETS[@]}"
echo "Cleaned ${#TARGETS[@]} path(s), ${total} MB freed."

hints=()
(( WANT_SIM ))  || hints+=("--sim for simulator traces")
(( WANT_JOBS )) || hints+=("--jobs for appliance job metadata")
(( WANT_LOGS )) || hints+=("--logs for root logs")
if (( ${#hints[@]} )); then
    printf 'Kept by default: %s\n' "$(IFS=', '; echo "${hints[*]}")"
fi
