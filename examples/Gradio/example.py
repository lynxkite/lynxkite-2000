from lynxkite_core.ops import op
import gradio as gr


def flip_text(x):
    return x[::-1]


@op("LynxKite Graph Analytics", "Gradio example", view="gradio")
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
