from lynxkite_core.ops import op
from lynxkite_graph_analytics import Bundle
import gradio as gr


def flip_text(x):
    return x[::-1]


with open("Gradio/gradio.svg") as f:
    gradio_icon = f.read()


@op("LynxKite Graph Analytics", "Gradio example", view="gradio", icon=gradio_icon)
def gradio_example():
    with gr.Blocks() as demo:
        gr.Markdown("""
            # Flip Text!
            Start typing below to see the output.
            """)
        input = gr.Textbox(placeholder="Flip this text")
        output = gr.Textbox()
        input.change(fn=flip_text, inputs=input, outputs=output)
    return demo


@op("LynxKite Graph Analytics", "Gradio DataFrame", view="gradio", icon=gradio_icon)
def gradio_df(bundle: Bundle):
    with gr.Blocks() as demo:
        for k in bundle.dfs:
            gr.Markdown(f"## {k}")
            gr.Dataframe(value=bundle.dfs[k])
    return demo
