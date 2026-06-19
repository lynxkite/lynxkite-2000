import os
import sys
import itertools
import pkgutil
import importlib

from lynxkite_core import ops


def get_box_skills():
    box_skills = []  # each box is a mini skill

    def _render_param(param):
        return f"{param['name']}=<{param['name']}_{'value' if param['kw'] else 'variable'}>"

    for _env, catalog in ops.CATALOGS.items():
        for op_name, op in catalog.items():
            name = op_name.split(">")[-1].strip()
            docstring = op.doc
            doc = {d["kind"]: d for d in docstring} if docstring else {}
            output_docs = doc["returns"]["value"] if "returns" in doc else []
            output_docs = {p["name"]: p["description"] for p in output_docs}
            text_doc = doc["text"]["value"] if "text" in doc else ""
            short_description = ""
            for line in text_doc.splitlines():
                if line.strip():
                    short_description = line.strip()
                    break
            skill = {
                "id": name.replace(" ", "-").replace("/", "-").replace("–", "-").lower(),
                "name": name,
                "description": short_description if short_description else name,
                "long_description": text_doc,
                "parameters": [],
                "usage": "",
                "returns": [(o.name, o.type, output_docs.get(o.name, "")) for o in op.outputs],
                "package": ".".join(op.python_function_name.split(".")[:-1])
                if op.python_function_name
                else "",
            }
            paramdocs = doc["parameters"]["value"] if "parameters" in doc else []
            paramdocs = {p["name"]: p["description"] for p in paramdocs}
            for argtype in [op.params, op.inputs]:
                for i, arg in enumerate(argtype):
                    param = {
                        "name": arg.name,
                        "type": arg.type,
                        "default": arg.default if hasattr(arg, "default") else None,
                        "kw": argtype == op.params,
                        "description": paramdocs.get(arg.name, ""),
                    }
                    skill["parameters"].append(param)
            skill["usage"] = (
                f"{op.python_function_name}({', '.join([_render_param(param) for param in skill['parameters']])})"
            )
            box_skills.append(skill)
    return box_skills


def create_box_description(skill) -> str:
    desc = f"""
**{skill["name"]}:**
{skill["long_description"]}
parameters:
{os.linesep.join([f"  - {param['name']}: {param['type'] or '?'} = {param['default'] or '?'} --{param['description'] or '?'}" for param in skill["parameters"]])}

returns:
{os.linesep.join([f"  - {ret[0]}: {ret[1] or '?'} - {ret[2] or '?'}." for ret in skill["returns"]])}

usage:
output_variable = {skill["usage"]}"""
    return desc


def create_package_skill(package, skills) -> tuple[str, str]:
    if len(skills) == 1:
        skill_name = skills[0]["id"]
        skill_desc = skills[0]["description"]
    else:
        skill_name = package.replace("_", "-").replace(".", "-").lower()
        skill_desc = f"Collection of operations - {', '.join(s['name'] for s in skills)}"
    content = f"""---
name: {skill_name}
description: {skill_desc}
---
{os.linesep.join([create_box_description(s) for s in skills])}
"""
    return skill_name, content


def _detect_certain_plugins(plugins_to_load: list[str]):
    plugins = {}
    for _, name, _ in pkgutil.iter_modules():
        if (
            name.startswith("lynxkite_")
            and name != "lynxkite_app"
            and name != "lynxkite_core"
            and name != "lynxkite_mcp"
            and name in plugins_to_load
        ):
            plugins[name] = importlib.import_module(name)
            if hasattr(plugins[name], "register_ops"):
                plugins[name].register_ops()


def create_skills_from_catalog(
    output_path: str = "./.agents/skills", changed_files: list[str] | None = None
):
    if changed_files:
        # assuming plugin name is the first part of the path, and - can be replaced with _ in the module name
        changed_plugins = list({f.split("/")[0].replace("-", "_") for f in changed_files})
        _detect_certain_plugins(changed_plugins)
    else:
        ops.detect_plugins()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    box_skills = get_box_skills()
    package_skills = itertools.groupby(box_skills, key=lambda s: s["package"])
    for package, skills in package_skills:
        skills = list(skills)
        name, content = create_package_skill(package, skills)
        skill_file_path = os.path.join(output_path, f"{name}")
        if not os.path.exists(skill_file_path):
            os.makedirs(skill_file_path)
        with open(os.path.join(skill_file_path, "SKILL.md"), "w") as f:
            f.write(content.strip() + os.linesep)  # Add a newline at the end for proper formatting


if __name__ == "__main__":
    if len(sys.argv) == 2:
        output_path = sys.argv[1]
        create_skills_from_catalog(output_path)
    elif len(sys.argv) > 2:
        output_path = sys.argv[1]
        create_skills_from_catalog(output_path, sys.argv[2:])
    else:
        create_skills_from_catalog()
