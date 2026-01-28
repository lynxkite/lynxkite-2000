import gradio as gr
import openai
import os
import urllib.parse
from . import python_to_workspace

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(message, history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        stream=True,
    )
    partial_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_response += chunk.choices[0].delta.content
            yield partial_response


converter = None


def sync_workspace(project):
    global converter
    os.makedirs(f"lynxkite-vibe/{project}", exist_ok=True)
    converter = python_to_workspace.CodeConverter(
        f"lynxkite-vibe/{project}/main.py", f"examples/LynxKite Vibe/{project}"
    )
    converter.start()
    project_encoded = urllib.parse.quote(project)
    return f"LynxKite workspace:\n## ➡️ [LynxKite Vibe/{project}](http://localhost:5173/edit/LynxKite%20Vibe/{project_encoded}/workspace.lynxkite.json)"


def main():
    with gr.Blocks() as demo:
        gr.ChatInterface(
            fn=chat,
            title="Vibe with LynxKite",
            description="Ask an agent to solve a data processing task. Review the solution as a LynxKite workspace.",
            examples=[
                "Download a list of Pokemon. Aggregate them by type. Create a histogram of the top 20 types. For the top 3 types create another chart that shows their count across different generations.",
                "Download a dataset containing height, weight, and blood type for a sample population. Analyze the data to find correlations between these attributes.",
                "Download the kjappelbaum/chemnlp-mp-magnetization dataset from Hugging Face if it's not already downloaded. Train a tiny transformer model for predicting magnetization from the molecular formula. Plot the training and validation loss curves. List the top 10 molecules with the highest test error in a table.",
            ],
        )
        project = gr.Textbox(label="Project", value="experiments/experiment 1")
        ws_link = gr.Markdown()
        gr.on([demo.load, project.change], inputs=[project], outputs=[ws_link], fn=sync_workspace)

    demo.launch()


if __name__ == "__main__":
    main()
