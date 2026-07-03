import os
import sys
import inspect
from lynxkite_core import ops


def _render_param(param):
    return f"{param['name']}=<{param['name']}_{'value' if param['kw'] else 'variable'}>"


class BoxSkill:
    def __init__(self, op_name: str, op: ops.Op):
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
        self.id = name.replace(" ", "-").replace("/", "-").replace("–", "-").lower()
        self.name = name
        self.short_description = short_description if short_description else name
        self.long_description = text_doc
        self.parameters = []
        self.usage = ""
        self.returns = [
            (o.name, o.type, output_docs.get(o.name, "")) for o in op.outputs
        ]
        self.package = (
            ".".join(op.python_function_name.split(".")[:-1])
            if op.python_function_name
            else ""
        )
        self.function_name = (
            op.python_function_name.split(".")[-1] if op.python_function_name else ""
        )
        self.code = inspect.getsource(op.func) if op.func else ""
        self.placeholder_function_name = getattr(op, "placeholder_function_name", False)
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
                self.parameters.append(param)
        self.usage = f"{op.python_function_name}({', '.join([_render_param(param) for param in self.parameters])})"

    def get_description(self) -> str:
        """For LynxKite boxes, the source code, for exteranal boxes, the parameters and usage are included in the description."""
        if "lynxkite" in self.package:
            desc = [
                f"**{self.name}:**",
                self.long_description,
                "```python",
                self.code,
                "```",
            ]
            custom_types = [
                (p["name"], p["type"])
                for p in self.parameters
                if "typing" in str(p["type"])
            ]
            if custom_types:
                desc.append("Custom types:")
                for name, typ in custom_types:
                    desc.append(f"  - {name}: {typ}")
        else:
            desc = [
                f"**{self.name}:**",
                self.long_description,
                "parameters:",
                *[
                    f"  - {param['name']}: {param['type'] or '?'} = {param['default'] or '?'} --{param['description'] or '?'}"
                    for param in self.parameters
                ],
                "returns:",
                *[
                    f"  - {ret[0]}: {ret[1] or '?'} - {ret[2] or '?'}."
                    for ret in self.returns
                ],
                "usage:",
                f"  output_variable = {self.usage}",
            ]
        return os.linesep.join(desc)

    def get_short_description(self) -> str:
        desc = [f"**{self.name}:**", f"usage: {self.usage}"]
        if not self.placeholder_function_name:
            desc.append(
                f"for detailed information, see references/{self.function_name or self.id}.md"
            )
        return os.linesep.join(desc)


def create_skills_from_catalog(
    output_path: str = "./.agents/skills", changed_files: list[str] | None = None
):
    if changed_files:
        regenerate = False
        # assuming plugin name is the first part of the path, and - can be replaced with _ in the module name
        for fn in changed_files:
            with open(fn, "r") as f:
                if "@op" in f.read():
                    regenerate = True
                    break
        if not regenerate:
            return  # if no changed files contain @op, we don't generate any skills
    ops.detect_plugins()
    output_path = os.path.join(output_path, "use-lynxkite-boxes")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "references")):
        os.makedirs(os.path.join(output_path, "references"))
    box_skills = []
    for _env, catalog in ops.CATALOGS.items():
        for op_name, op in catalog.items():
            box_skills.append(BoxSkill(op_name, op))
    box_skills.sort(key=lambda x: (x.package, x.id))
    main_skill_file = f"""---
name: use-lynxkite-boxes
description: Use the boxes already defined in LynxKite.
---
## Inserting boxes into the workspace:
1. inspect the available boxes and choose the one that fits your needs
2. check their detailed documentation in the references folder
3. insert the box into your workspace by calling the corresponding function in `workspace.py` with the appropriate parameters
4. double-check the parameters and their types with the box's documentation in the references folder

## Available boxes
The following boxes are available for use in your workflows.
Each box corresponds to a specific operation or function that can be used to build your workflow.
For detailed information on each box, please refer to the individual box documentation in the references folder.
Always check the references before using the box, and pay close attention to the parameters and their types.

{(os.linesep * 2).join([s.get_short_description() for s in box_skills])}
"""
    with open(os.path.join(output_path, "SKILL.md"), "w") as f:
        f.write(
            main_skill_file.strip() + os.linesep
        )  # Add a newline at the end for proper formatting
    for s in box_skills:
        if s.placeholder_function_name:
            continue  # skip functions where there's no real function, e.g. for lambdas or no_op
        with open(
            os.path.join(output_path, "references", f"{s.function_name or s.id}.md"),
            "w",
        ) as f:
            f.write(
                s.get_description().strip() + os.linesep
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
