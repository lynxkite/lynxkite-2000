from lynxkite_assistant import workspace_backend, assistant
from pathlib import Path
from lynxkite_core import ops
import os
import asyncio

samples = [
    ("lynxkite-assistant/tests/files/blank.lynxkite.json", "Hi"),
    (
        "lynxkite-assistant/tests/files/blank.lynxkite.json",
        "Import dataset: https://storage.googleapis.com/lynxkite_public_data/airlines.graphml.gz and create visualizations for most outgoing and incoming flights by airports.",
    ),
    (
        "examples/Basic examples/Airlines demo.lynxkite.json",
        "Off the imported graph data, add a new branch that lists all the airports in a table.",
    ),
    (
        "examples/Basic examples/Image processing.lynxkite.json",
        "After the grayscale step, flip the result vertically and show it in a new preview.",
    ),
    (
        "examples/Basic examples/Image processing.lynxkite.json",
        "Add a vertical flip between the Blur and the To grayscale steps.",
    ),
    ("examples/Basic examples/Image processing.lynxkite.json", "Delete the Blur box."),
    (
        "examples/Basic examples/Image processing.lynxkite.json",
        "Set the parameter of Blur to 10.",
    ),
    (
        "examples/Basic examples/Image processing.lynxkite.json",
        "Add a custom box that writes a text caption onto the image and place it before final preview.",
    ),
    (
        "examples/Basic examples/Image processing.lynxkite.json",
        ' Fork "To grayscale" into a custom box with a threshold parameter.',
    ),
]

ops.detect_plugins()
ops.user_script_root = Path()
workspace_backend.crdt = None  # type: ignore
assistant.crdt = None  # type: ignore
dir = Path("examples/generated_samples")


def init_files(i, og_filename):
    file = dir / Path(f"sample_{i}_" + Path(og_filename).name)
    os.makedirs(dir, exist_ok=True)
    file.touch(exist_ok=True)
    with open(og_filename) as f, open(file, "w") as f2:
        f2.write(f.read())
    return file


def save_boxes_py(i):
    if not os.path.exists(dir / "boxes.py"):
        return
    file = dir / Path(f"sample_{i}_boxes.py")
    file.touch(exist_ok=True)
    with open(dir / "boxes.py", "r") as f, open(file, "w") as f2:
        f2.write(f.read())
    os.remove(dir / "boxes.py")


async def test_assistant_stream():
    for i, (og_filename, prompt) in enumerate(samples):
        file = init_files(i, og_filename)
        req = assistant.AssistantCompletionRequest(
            workspace=str(file),
            messages=[assistant.AssistantMessage(role="user", content=prompt)],
        )
        print(
            f"--------------------------------------- Sample {i}: {file.name} ---------------------------------------"
        )
        print("User:")
        print(prompt)
        resp = await assistant.assistant_stream(req, skill_root=".agents/skills")
        resp_chunks = []
        async for chunk in resp.body_iterator:
            resp_chunks.append(chunk)
        print("Assistant:")
        print("".join(resp_chunks))
        save_boxes_py(i)


asyncio.run(test_assistant_stream())
