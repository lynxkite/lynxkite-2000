import ast
import os


def create_skill_from_op(name: str, node: ast.FunctionDef, python_path="") -> dict:
    docstring = ast.get_docstring(node) or ""
    description = ""
    for line in docstring.splitlines():
        if line.strip():
            description = line.strip()
            break
    skill = {
        "name": name,
        "description": description or name,
        "long_description": docstring,
        "parameters": [],
        "usage": "",
    }

    def _decode_atrr(node, default=None):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{_decode_atrr(node.value)}.{node.attr}"
        else:
            return default

    for argtype, default in [
        (node.args.args, node.args.defaults),
        (node.args.kwonlyargs, node.args.kw_defaults),
    ]:
        for i, arg in enumerate(argtype):
            param = {
                "name": arg.arg,
                "type": _decode_atrr(arg.annotation, "Any"),
                "default": _decode_atrr(default[i]) if default and i < len(default) else None,
                "kw": argtype == node.args.kwonlyargs,
            }
            skill["parameters"].append(param)

    def _render_param(param):
        return f"{param['name']}=<{param['name']}_{'value' if param['kw'] else 'variable'}>"

    skill["usage"] = (
        f"{python_path + node.name}({', '.join([_render_param(param) for param in skill['parameters']])})"
    )
    return skill


def extract_ops_from_file(file_path: str):
    with open(file_path, "r") as f:
        code = f.read()
    tree = ast.parse(code)

    def get_op_name(node: ast.FunctionDef) -> str | None:
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "op"
            ):
                return (
                    decorator.args[0].value.replace(" ", "_").replace("/", "-")
                    if decorator.args
                    else node.name
                )
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = get_op_name(node)
            if name is not None:
                yield name, node


def create_skills_from_file(file_path: str, output_path: str = "./.agents/skills"):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for name, node in extract_ops_from_file(file_path):
        python_path = file_path.replace("/", ".").replace("-", "_")[:-2]
        skill = create_skill_from_op(name, node, python_path=python_path)
        content = f"""
---
name: {skill["name"]}
description: {skill["description"]}
---

{skill["long_description"]}

parameters:
{os.linesep.join([f"  - {param['name']}: {param['type']} = {param['default']}" for param in skill["parameters"]])}

usage:
output_variable = {skill["usage"]}

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.

        """
        if not os.path.exists(output_path + "/" + name):
            os.makedirs(output_path + "/" + name)
        with open(output_path + "/" + name + "/SKILL.md", "w") as f:
            f.write(content)
        print("created skill:", name)


def create_skills_from_directory(directory_path: str, output_path: str = "./.agents/skills"):
    for root, _dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                create_skills_from_file(os.path.join(root, file), output_path)


def main(argv):
    if len(argv) < 2:
        print("Usage: python generate_skill_from_file.py <file_path> [output_path]")
        return
    file_path = argv[1]
    output_path = argv[2] if len(argv) > 2 else "./.agents/skills"
    print(file_path, output_path, os.path.isdir(file_path))
    if os.path.isdir(file_path):
        create_skills_from_directory(file_path, output_path)
    else:
        create_skills_from_file(file_path, output_path)


if __name__ == "__main__":
    import sys

    main(sys.argv)
