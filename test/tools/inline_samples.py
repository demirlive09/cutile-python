# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import sys
import pathlib
import shutil
import warnings
from typing import Dict


ROOT = pathlib.Path(__file__).resolve().parents[2]
KERNELS_DIR = ROOT / "test" / "kernels"
SAMPLES_DIR = ROOT / "samples"
SAMPLES_TEMPLATES_DIR = SAMPLES_DIR / "templates"


if sys.version_info >= (3, 12, 0, 0, 0):
    TypeAlias = ast.TypeAlias
else:
    class TypeAlias:
        pass


def _used_names_in_function(fn: ast.FunctionDef) -> set[str]:
    """Collect names read in body, annotations, decorators, defaults;
       also include attribute bases like `ct` in `ct.load`."""
    used: set[str] = set()
    locals_: set[str] = {a.arg for a in (fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs)}
    if fn.args.vararg:
        locals_.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        locals_.add(fn.args.kwarg.arg)

    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name):
            if isinstance(n.ctx, ast.Load):
                used.add(n.id)
            elif isinstance(n.ctx, ast.Store):
                locals_.add(n.id)

        def visit_Attribute(self, n: ast.Attribute):
            # capture the base name in ct.foo
            if isinstance(n.value, ast.Name):
                used.add(n.value.id)
            self.generic_visit(n)

        def visit_arg(self, n: ast.arg):
            if n.annotation:
                self.visit(n.annotation)

        def visit_FunctionDef(self, n: ast.FunctionDef):
            for d in n.decorator_list:
                self.visit(d)
            for a in n.args.posonlyargs + n.args.args + n.args.kwonlyargs:
                self.visit(a)
            if n.returns:
                self.visit(n.returns)
            for s in n.body:
                self.visit(s)

    V().visit(fn)
    used.discard(fn.name)
    used.difference_update(locals_)
    return used


def get_helper_nodes(
    node: ast.FunctionDef,
    all_func_nodes: Dict[str, ast.FunctionDef],
    all_alias_nodes: Dict[str, TypeAlias | ast.Assign | ast.AnnAssign],
    all_import_nodes: Dict[str, ast.ImportFrom | ast.Import],
    func_nodes_to_add: Dict[str, ast.FunctionDef],
    alias_nodes_to_add: Dict[str, TypeAlias | ast.Assign | ast.AnnAssign],
    import_nodes_to_add: Dict[str, ast.ImportFrom | ast.Import],
) -> None:
    used_names = _used_names_in_function(node)
    for name in used_names:
        if name in all_func_nodes:
            func_nodes_to_add[name] = all_func_nodes[name]
            get_helper_nodes(
                all_func_nodes[name],
                all_func_nodes, all_alias_nodes, all_import_nodes,
                func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add
            )
        elif name in all_alias_nodes:
            alias_nodes_to_add[name] = all_alias_nodes[name]
        elif name in all_import_nodes:
            import_nodes_to_add[name] = all_import_nodes[name]
        else:
            warnings.warn(f"Unknown name: {name}, it will not be inlined to the sample.")


def get_all_nodes(tree: ast.Module) -> tuple:
    all_func_nodes: Dict[str, ast.FunctionDef] = {}
    all_alias_nodes: Dict[str, TypeAlias | ast.Assign | ast.AnnAssign] = {}
    all_import_nodes: Dict[str, ast.ImportFrom | ast.Import] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            all_func_nodes[node.name] = node
        elif isinstance(node, TypeAlias):
            all_alias_nodes[node.name.id] = node
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            all_alias_nodes[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            all_alias_nodes[node.target.id] = node
        elif isinstance(node, ast.Import):
            for a in node.names:
                intro = a.asname or a.name.split(".", 1)[0]
                all_import_nodes[intro] = node
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                intro = a.asname or a.name
                all_import_nodes[intro] = node
    return all_func_nodes, all_alias_nodes, all_import_nodes


def find_nodes_to_add(tree: ast.Module, import_names: list[str]) -> tuple:
    all_func_nodes, all_alias_nodes, all_import_nodes = get_all_nodes(tree)
    func_nodes_to_add: Dict[str, ast.FunctionDef] = {}
    alias_nodes_to_add: Dict[str, TypeAlias | ast.Assign | ast.AnnAssign] = {}
    import_nodes_to_add: Dict[str, ast.ImportFrom | ast.Import] = {}
    for name in import_names:
        node = all_func_nodes[name]
        func_nodes_to_add[node.name] = node
        get_helper_nodes(
            node,
            all_func_nodes, all_alias_nodes, all_import_nodes,
            func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add
        )
    return func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add


def get_kernels_and_helpers_content(import_node: ast.ImportFrom, dst_tree: ast.Module) -> list[str]:
    rel = pathlib.Path(*import_node.module.split(".")).with_suffix(".py")
    with open(ROOT / rel, "r") as f:
        code = f.read()
    code_lines = code.splitlines()
    tree = ast.parse(code)
    func_nodes_to_add, alias_nodes_to_add, import_nodes_to_add = find_nodes_to_add(
        tree, [name.name for name in import_node.names]
    )
    _, dst_alias_nodes, dst_import_nodes = get_all_nodes(dst_tree)
    # Dedup alias and import nodes
    alias_nodes_to_add = {
        k: v for k, v in alias_nodes_to_add.items() if k not in dst_alias_nodes
    }
    import_nodes_to_add = {
        k: v for k, v in import_nodes_to_add.items() if k not in dst_import_nodes
    }

    res = []
    # Add codes for import nodes and alias nodes
    if import_nodes_to_add:
        for node in sorted(import_nodes_to_add.values(), key=lambda x: x.lineno):
            res.extend(code_lines[node.lineno-1:node.end_lineno])
        res.extend(["", ""])
    if alias_nodes_to_add:
        for node in sorted(alias_nodes_to_add.values(), key=lambda x: x.lineno):
            res.extend(code_lines[node.lineno-1:node.end_lineno])
        res.extend(["", ""])
    # Add codes for function nodes
    for i, node in enumerate(sorted(func_nodes_to_add.values(), key=lambda x: x.lineno)):
        if i > 0:
            res.extend(["", ""])
        if len(node.decorator_list) == 1:
            # Kernel function
            res.extend(code_lines[node.decorator_list[0].lineno-1:node.end_lineno])
        else:
            # Helper function
            res.extend(code_lines[node.lineno-1:node.end_lineno])
    return res


def _extend_with_empty_lines(lines: list[str]) -> None:
    if not lines or lines[-1] != "":
        lines.extend(["", ""])
    elif len(lines) == 1 or lines[-2] != "":
        lines.append("")


def replace_kernel_content(py: pathlib.Path, prefix: str) -> tuple[bool, list[str]]:
    """Replace the kernel import lines with the content of the imported modules."""
    with open(py, "r") as f:
        code = f.read()
    tree = ast.parse(code)
    code_lines = code.splitlines()

    replace_map = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module.startswith(prefix):
                replace_map[node.lineno] = (
                    node.end_lineno, get_kernels_and_helpers_content(node, tree)
                )
    new_code_lines = []
    if replace_map:
        changed = True
        i = 0
        while i < len(code_lines):
            line_no = i + 1
            if line_no in replace_map:
                _extend_with_empty_lines(new_code_lines)
                new_code_lines.extend(replace_map[line_no][1])
                i = replace_map[line_no][0]
            else:
                new_code_lines.append(code_lines[i])
                i += 1
    return changed, new_code_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default="test.kernels",
        type=str,
        help="Module prefix to inline from (default: test.kernels)"
    )
    parser.add_argument("--template-dir", type=pathlib.Path, default=SAMPLES_TEMPLATES_DIR)
    args = parser.parse_args()

    for py in args.template_dir.rglob("*.py"):
        if py.name == "__init__.py":
            continue
        changed, out = replace_kernel_content(py, args.prefix)
        if changed:
            sample_path = SAMPLES_DIR / py.relative_to(SAMPLES_TEMPLATES_DIR)
            sample_path.write_text("\n".join(out) + "\n", encoding="utf-8")
            print(f"Updated {sample_path.relative_to(ROOT)}")

    # Copy the autotuner.py to the samples directory
    autotuner_path = ROOT / "test" / "autotuner" / "autotuner.py"
    samples_autotuner_path = SAMPLES_DIR / "utils" / "autotuner.py"
    shutil.copy(autotuner_path, samples_autotuner_path)
    print(f"Copied {autotuner_path.relative_to(ROOT)} to "
          f"{samples_autotuner_path.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
