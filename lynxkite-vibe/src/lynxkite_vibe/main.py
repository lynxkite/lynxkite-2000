import gradio as gr
import openai
import os
import json
import urllib.parse
from . import python_to_workspace

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
current_project = None
converter = None


def read_script():
    assert current_project
    dir_path = f"lynxkite-vibe/{current_project}"
    os.makedirs(dir_path, exist_ok=True)
    path = f"{dir_path}/main.py"
    if not os.path.exists(path):
        return "<empty file>"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


read_script_tool = {
    "type": "function",
    "function": {
        "name": "read_script",
        "description": "Read the content of the Python script we are writing.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def write_script(content):
    assert current_project
    dir_path = f"lynxkite-vibe/{current_project}"
    os.makedirs(dir_path, exist_ok=True)
    path = f"{dir_path}/main.py"
    print(f"Writing to {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


write_script_tool = {
    "type": "function",
    "function": {
        "name": "write_script",
        "description": "Write content to the Python script we are writing.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to write to the Python script.",
                },
            },
            "required": ["content"],
        },
    },
}

TOOLS = [read_script_tool, write_script_tool]

TOOL_FUNCTIONS = {
    "read_script": read_script,
    "write_script": write_script,
}

SYSTEM_PROMPT = """
You are an AI agent that generates Python scripts for data processing and visualization tasks.
Use the provided tools to read the current script content and write updates to it.
Structure the Python code as a series of functions. Create a single `main()` function
that calls these functions to perform the overall task. Each line in `main()` should be
a single function call, optionally assigning the result to a variable.

For example:

```python
def load_data():
    # code to load data
def process_data(data, learning_rate: float):
    # code to process data
def plot_results(processed_data, color: str):
    # code to plot results

def main():
    data = load_data()
    processed = process_data(data, learning_rate=0.01)
    plot_results(processed, color="blue")
```
""".strip()


def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    while True:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
            tools=TOOLS,
        )

        # Accumulate streaming response
        partial_response = ""
        tool_calls = {}

        for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta

            # Handle content streaming
            if delta.content is not None:
                partial_response += delta.content
                yield partial_response

            # Accumulate tool call deltas
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id,
                            "function": {"name": tc.function.name, "arguments": ""},
                            "type": "function",
                        }
                    if tc.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        # If no tool calls, we're done
        if not tool_calls:
            break

        # Add assistant message with tool calls to messages
        assistant_message = {"role": "assistant", "tool_calls": list(tool_calls.values())}
        if partial_response:
            assistant_message["content"] = partial_response
        messages.append(assistant_message)

        # Execute tool calls and add results
        for tc in tool_calls.values():
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            print("Executing tool:", fn_name, "with args:", fn_args)
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = f"Unknown function: {fn_name}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result) if result is not None else "success",
                }
            )


def sync_workspace(project):
    global converter, current_project
    current_project = project
    os.makedirs(f"lynxkite-vibe/{project}", exist_ok=True)
    converter = python_to_workspace.CodeConverter(
        f"lynxkite-vibe/{project}/main.py", f"examples/LynxKite Vibe/{project}"
    )
    converter.start()
    project_encoded = urllib.parse.quote(project)
    url = f"http://localhost:5173/edit/LynxKite%20Vibe/{project_encoded}/workspace.lynxkite.json"
    return f"LynxKite workspace:\n## ➡️ [LynxKite Vibe/{project}]({url})"


def format_example(s: str) -> str:
    return " ".join(s.split())


def main():
    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=chat,
            title="Vibe with LynxKite",
            description="Ask an agent to solve a data processing task. Review the solution as a LynxKite workspace.",
            examples=[
                format_example("""
                Download a list of Pokemon. Aggregate them by type.
                Create a histogram of the top 20 types.
                For the top 3 types create another chart that shows their count across different generations.
                """),
                format_example("""
                Download a dataset containing height, weight, and blood type for a sample population.
                Analyze the data to find correlations between these attributes.
                """),
                format_example("""
                Download the kjappelbaum/chemnlp-mp-magnetization dataset from Hugging Face if it's not already downloaded.
                Train a tiny transformer model for predicting magnetization from the molecular formula.
                Plot the training and validation loss curves.
                List the top 10 molecules with the highest test error in a table.
                """),
            ],
        )
        project = gr.Textbox(label="Project", value="experiments/experiment 1")
        ws_link = gr.Markdown()
        gr.on([demo.load, project.change], inputs=[project], outputs=[ws_link], fn=sync_workspace)

    demo.launch()


if __name__ == "__main__":
    main()
