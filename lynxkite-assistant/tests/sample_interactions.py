"""Automatically runs some sample prompts on the demo workspaces and saves the results to files in examples/generated_samples.

Run from the root of the repository with `python lynxkite-assistant/tests/sample_interactions.py` to generate all interactions.
You can also run it with `python lynxkite-assistant/tests/sample_interactions.py skip` to skip samples that already have a generated file in examples/generated_samples.
"""

from lynxkite_core import ops
from lynxkite_assistant import workspace_backend, assistant
from pathlib import Path
import os
import sys
import asyncio


class SampleInput:
    def __init__(self, filename: str, prompt: str, title: str = ""):
        self.filename = filename
        self.prompt = prompt
        if not title:
            self.title = prompt[:30]
            if len(prompt) > 30:
                self.title += "..."
        else:
            self.title = title


samples = [
    SampleInput(
        "lynxkite-assistant/tests/files/blank.lynxkite.json",
        "Hi",
        "Blank workspace, simple greeting",
    ),
    SampleInput(
        "lynxkite-assistant/tests/files/blank.lynxkite.json",
        "Import dataset: https://storage.googleapis.com/lynxkite_public_data/airlines.graphml.gz and create visualizations for most outgoing and incoming flights by airports.",
        "Blank workspace pipeline build",
    ),
    SampleInput(
        "examples/Basic examples/Airlines demo.lynxkite.json",
        "Off the imported graph data, add a new branch that lists all the airports in a table.",
        "Insert Lynxkite node at the end of a branch",
    ),
    SampleInput(
        "examples/Basic examples/Image processing.lynxkite.json",
        "After the grayscale step, flip the result vertically and show it in a new preview.",
        "Insert Lynxkite node in a new branch",
    ),
    SampleInput(
        "examples/Basic examples/Image processing.lynxkite.json",
        "Add a vertical flip between the Blur and the To grayscale steps.",
        "Insert Lynxkite node in middle + rewire safely",
    ),
    SampleInput(
        "examples/Basic examples/Image processing.lynxkite.json",
        "Delete the Blur box.",
        "Delete one box from the middle + rewire safely",
    ),
    SampleInput(
        "examples/Basic examples/Image processing.lynxkite.json",
        "Set the parameter of Blur to 10.",
        "Edit one parameter of a box",
    ),
    SampleInput(
        "examples/Basic examples/Image processing.lynxkite.json",
        "Add a custom box that writes a text caption onto the image and place it before final preview.",
        "Create new local custom box",
    ),
    SampleInput(
        "examples/Basic examples/Image processing.lynxkite.json",
        'Fork "To grayscale" into a custom box with a threshold parameter.',
        "Create custom box based on Lynxkite box",
    ),
    SampleInput(
        "lynxkite-assistant/tests/files/disorganized.lynxkite.json",
        "Organize the layout of boxes.",
    ),
    SampleInput(
        "lynxkite-assistant/tests/files/disorganized.lynxkite.json",
        "Clean up orphan boxes that are not connected.",
    ),
    SampleInput(
        "lynxkite-assistant/tests/files/N16_no_comments.lynxkite.json",
        "Understand the workspace and add comments to explain functionality.",
    ),
    SampleInput(
        "lynxkite-assistant/tests/files/N16_broken.lynxkite.json",
        "Fix the broken workspace",
    ),
    SampleInput(
        "examples/Peter's lessons/soccer.lynxkite.json",
        "Reorganize the boxes so the flow is going top to bottom instead of left to right.",
    ),
]

dir = Path("examples/generated_samples")


def save_boxes_py(i):
    if not os.path.exists(dir / "boxes.py"):
        return
    file = dir / Path(f"sample_{i}_boxes.py")
    file.touch(exist_ok=True)
    with open(dir / "boxes.py", "r") as f, open(file, "w") as f2:
        f2.write(f.read())
    os.remove(dir / "boxes.py")


async def test_assistant_stream(skip_existing=False):
    m = len(str(len(samples) - 1))
    for i, inp in enumerate(samples):
        file = dir / Path(
            f"sample_{str(i).zfill(m)}_"
            + "".join(c if c.isalnum() else "_" for c in inp.title)
            + ".lynxkite.json"
        )
        os.makedirs(dir, exist_ok=True)
        if skip_existing and file.exists():
            print(f"Skipping sample {i}: {inp.title} (file exists)")
            continue
        file.touch(exist_ok=True)
        with open(inp.filename) as f, open(file, "w") as f2:
            f2.write(f.read())
        req = assistant.AssistantCompletionRequest(
            workspace=str(file),
            messages=[assistant.AssistantMessage(role="user", content=inp.prompt)],
        )
        print(
            f"--------------------------------------- Sample {i}: {inp.title} ---------------------------------------"
        )
        print("User:")
        print(inp.prompt)
        resp = await assistant.assistant_stream(req, skill_root=".agents/skills")
        resp_chunks = []
        async for chunk in resp.body_iterator:
            resp_chunks.append(chunk)
        print("Assistant:")
        print("".join(resp_chunks))
        save_boxes_py(i)


if __name__ == "__main__":
    ops.detect_plugins()
    ops.user_script_root = Path()
    workspace_backend.crdt = None  # type: ignore
    assistant.crdt = None  # type: ignore
    if len(sys.argv) > 1 and sys.argv[1] == "skip":
        asyncio.run(test_assistant_stream(skip_existing=True))
    else:
        asyncio.run(test_assistant_stream())
