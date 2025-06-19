from llama2_medical.model.infer import generate_llama2_answer
import gradio as gr

def main():
    iface = gr.Interface(
        fn=generate_llama2_answer,
        inputs=gr.Textbox(lines=3, label="Enter your medical question"),
        outputs=gr.Textbox(label="Llama2 Medical Answer"),
        title="Llama2 Medical Q&A",
        description="Ask any medical question and get an answer from the fine-tuned Llama2 model."
    )
    iface.launch(share=True)  # share=True gives a public link in Colab

if __name__ == "__main__":
    main()
