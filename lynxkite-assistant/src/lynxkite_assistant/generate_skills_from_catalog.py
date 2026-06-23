import os
import sys
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
                "short_description": short_description if short_description else name,
                "long_description": text_doc,
                "parameters": [],
                "usage": "",
                "returns": [(o.name, o.type, output_docs.get(o.name, "")) for o in op.outputs],
                "package": ".".join(op.python_function_name.split(".")[:-1])
                if op.python_function_name
                else "",
                "function_name": op.python_function_name.split(".")[-1]
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


def create_short_box_description(skill) -> str:
    desc = f"""**{skill["name"]}:**
{skill["short_description"]}
for usage information, see references/{skill["function_name"] or skill["id"]}.md"""
    return desc


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
        if any(
            p.startswith("lynxkite_")
            and p != "lynxkite_app"
            and p != "lynxkite_core"
            and p != "lynxkite_mcp"
            for p in changed_plugins
        ):
            ops.detect_plugins()
        else:
            return
    else:
        ops.detect_plugins()
    output_path = os.path.join(output_path, "use-lynxkite-boxes")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "references")):
        os.makedirs(os.path.join(output_path, "references"))
    box_skills = get_box_skills()
    box_skills.sort(key=lambda x: (x["package"], x["id"]))
    main_skill_file = f"""---
name: use-lynxkite-boxes
description: Use the boxes already defined in LynxKite.
---
## Available boxes
The following boxes are available for use in your workflows.
Each box corresponds to a specific operation or function that can be used to build your workflow.
For detailed information on each box, please refer to the individual box documentation in the references folder.

{(os.linesep * 2).join([create_short_box_description(s) for s in box_skills])}
"""
    with open(os.path.join(output_path, "SKILL.md"), "w") as f:
        f.write(
            main_skill_file.strip() + os.linesep
        )  # Add a newline at the end for proper formatting
    for s in box_skills:
        with open(
            os.path.join(output_path, "references", f"{s['function_name'] or s['id']}.md"), "w"
        ) as f:
            f.write(
                create_box_description(s).strip() + os.linesep
            )  # Add a newline at the end for proper formatting


if __name__ == "__main__":
    """
    Usage examples:
    python generate_skills_from_catalog.py
    - This will generate skills for all plugins and save them under .agents/skills/ in the working directory.
    python generate_skills_from_catalog.py ./output_path
    - This will generate skills for all plugins and save them under the specified output_path.
    python generate_skills_from_catalog.py ./output_path changed_file1.py changed_file2.py
    - This will generate skills only for the plugins that have changed files and save them under the specified output_path.
      (this is mainly for the pre-commit hook)
    """
    if len(sys.argv) == 2:
        output_path = sys.argv[1]
        create_skills_from_catalog(output_path)
    elif len(sys.argv) > 2:
        output_path = sys.argv[1]
        create_skills_from_catalog(output_path, sys.argv[2:])
    else:
        create_skills_from_catalog()
