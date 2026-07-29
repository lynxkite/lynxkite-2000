def gather_multiline_comments(code: str) -> list[tuple[int, str]]:
    comments = []
    comment_lines = []
    next_line = 0
    for i, line in enumerate(code.splitlines()):
        if not line.strip().startswith("#!"):
            continue
        if next_line == i:
            comment_lines.append(line.strip()[2:].strip())
        else:
            if comment_lines:
                comments.append(
                    (next_line - len(comment_lines), "\n".join(comment_lines))
                )
            comment_lines = [line.strip()[2:].strip()]
        next_line = i + 1
    if comment_lines:
        comments.append((next_line - len(comment_lines), "\n".join(comment_lines)))
    return comments


def describe_schema(schema: dict) -> str:
    lines = [f"bundle with {len(schema['dataframes'])} dataframe(s)"]
    for df_name, info in sorted(schema["dataframes"].items()):
        cols = ", ".join(f"'{c}'" for c in info.get("columns", []))
        lines.append(f"'{df_name}' with columns {cols}")
    return "\n# ".join(lines)


def compare_dataframe_schemas(prev_schema: dict, new_schema: dict) -> str:
    changes = []
    prev_keys = set(prev_schema["dataframes"].keys())
    new_keys = set(new_schema["dataframes"].keys())
    all_keys = prev_keys.union(new_keys)
    if prev_keys.intersection(new_keys) == set():
        return f"new bundle: {describe_schema(new_schema)}"
    for key in sorted(all_keys):
        if key not in prev_schema["dataframes"]:
            cols = ", ".join(f"'{c}'" for c in new_schema["dataframes"][key]["columns"])
            changes.append(f"new dataframe '{key}' added with columns: {cols}")
        elif key not in new_schema["dataframes"]:
            changes.append(f"dataframe '{key}' was removed")
        else:
            prev_cols = set(prev_schema["dataframes"][key]["columns"])
            new_cols = set(new_schema["dataframes"][key]["columns"])
            added_cols = sorted(list(new_cols - prev_cols))
            removed_cols = sorted(list(prev_cols - new_cols))
            if added_cols:
                cols_str = ", ".join(f"'{c}'" for c in added_cols)
                changes.append(f"'{key}' now has additional column(s): {cols_str}")
            if removed_cols:
                cols_str = ", ".join(f"'{c}'" for c in removed_cols)
                changes.append(f"'{key}' had column(s) removed: {cols_str}")
    return "\n# ".join(changes) if changes else "no change"


def get_inp_op_metadata_comments(
    parent_op: list[dict], ip: list[dict], op: list[dict]
) -> tuple[str, str]:
    def is_df_schema(d: dict) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        return "dataframes" in d and isinstance(d["dataframes"], dict)

    def get_compact_summary(meta: dict) -> str:
        if not meta:
            return ""
        keys_str = ", ".join(meta.keys())
        return f"metadata with keys: {keys_str}"

    input_descriptions = []
    # Filter out empty dicts from the lists completely
    parent_ops_clean = [p for p in parent_op if p]
    ips_clean = [i for i in ip if i]
    ops_clean = [o for o in op if o]
    for incoming in ips_clean:
        if incoming in parent_ops_clean:
            continue
        if is_df_schema(incoming):
            input_descriptions.append(f"input: {describe_schema(incoming)}")
        elif incoming:
            input_descriptions.append(f"input: {get_compact_summary(incoming)}")
    input_comment = "\n# ".join(input_descriptions) if input_descriptions else ""

    output_descriptions = []
    if ops_clean:
        primary_op = ops_clean[0]
        if is_df_schema(primary_op):
            # Find the most relevant baseline to compare against (input or parent output)
            baseline = next((i for i in ips_clean if is_df_schema(i)), None)
            if not baseline:
                baseline = next((p for p in parent_ops_clean if is_df_schema(p)), None)
            if baseline:
                output_descriptions.append(
                    "output: " + compare_dataframe_schemas(baseline, primary_op)
                )
            else:
                output_descriptions.append(
                    f"output: new bundle: {describe_schema(primary_op)}"
                )
        else:
            output_descriptions.append(f"output: {get_compact_summary(primary_op)}")
    output_comment = "\n# ".join(output_descriptions) if output_descriptions else ""
    return input_comment, output_comment
