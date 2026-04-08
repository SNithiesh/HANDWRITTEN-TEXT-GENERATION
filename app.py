import os
import gradio as gr
from handwritten_rnn.generator import Generator

# Default paths — bundled in the repo under model/
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "model/best.pt")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "model/vocab.json")

# Load model once at startup
generator = None
load_error = None

try:
    generator = Generator(checkpoint_path=CHECKPOINT_PATH, vocab_path=VOCAB_PATH)
except Exception as e:
    load_error = str(e)


def generate_text(seed_text: str, num_chars: int, temperature: float) -> str:
    if load_error:
        return f"Model failed to load: {load_error}"
    if generator is None:
        return "Model not loaded."
    try:
        return generator.generate(
            seed_text=seed_text,
            num_chars=int(num_chars),
            temperature=float(temperature),
        )
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Handwritten Text Generation") as demo:
    gr.Markdown("# ✍️ Handwritten Text Generation")
    gr.Markdown(
        "A character-level RNN trained on the "
        "[IAM Handwriting Dataset](https://huggingface.co/datasets/Teklia/IAM-line). "
        "Type a seed phrase and let the model continue it in handwritten English style."
    )

    if load_error:
        gr.Markdown(f"> ⚠️ Model load error: `{load_error}`")

    with gr.Row():
        seed_text = gr.Textbox(
            value="The quick brown fox",
            label="Seed text",
            placeholder="Enter starting text...",
        )

    with gr.Row():
        num_chars = gr.Slider(
            minimum=50, maximum=1000, value=300, step=50,
            label="Number of characters to generate"
        )
        temperature = gr.Slider(
            minimum=0.1, maximum=2.0, value=0.5, step=0.1,
            label="Temperature (lower = more predictable, higher = more creative)"
        )

    generate_btn = gr.Button("Generate ✍️", variant="primary")
    output = gr.Textbox(label="Generated text", lines=8, interactive=False)

    generate_btn.click(
        generate_text,
        inputs=[seed_text, num_chars, temperature],
        outputs=output,
    )

    gr.Examples(
        examples=[
            ["The government announced", 300, 0.5],
            ["Once upon a time", 300, 0.7],
            ["In the early morning", 200, 0.4],
            ["", 300, 0.8],
        ],
        inputs=[seed_text, num_chars, temperature],
    )

if __name__ == "__main__":
    demo.launch()
