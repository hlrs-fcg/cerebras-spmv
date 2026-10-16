"""The worker's source text, for the tests that pin its structure.

cluster/worker_runtime.py is the entry point; the stages live in sibling modules (staged flat).  The
text is joined in execution order (CLI, host, mode hooks, device, report, helpers), so a test that
asserts "X appears before Y" still reads as the run does.
"""

from pathlib import Path

CLUSTER = Path(__file__).resolve().parent.parent / "cluster"
ORDER = ("worker_runtime", "worker_cli", "worker_host", "worker_stream", "worker_resident",
         "worker_modes", "worker_device", "worker_report", "worker_common", "worker_layout",
         "worker_timing")


def worker_files():
    return [CLUSTER / f"{n}.py" for n in ORDER]


def worker_source() -> str:
    return "\n".join(p.read_text() for p in worker_files())


def symbols_read() -> set:
    """Every device symbol the worker fetches with runner.get_id: the common tuple in run_device,
    the mode hooks' extra_symbol_names, and sync_pack."""
    import ast
    names = set()
    for path in worker_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "get_id":
                for a in node.args:
                    if isinstance(a, ast.Constant):
                        names.add(a.value)
            if isinstance(node, ast.FunctionDef) and node.name == "extra_symbol_names":
                names |= {c.value for c in ast.walk(node)
                          if isinstance(c, ast.Constant) and isinstance(c.value, str)
                          and not c.value.startswith(("Symbols", "vals:", "row_ids:"))
                          and c.value.isidentifier()}
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple) and any(
                    isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get_id"
                    for n in ast.walk(node)):
                names |= {e.value for e in node.iter.elts if isinstance(e, ast.Constant)}
    return names
