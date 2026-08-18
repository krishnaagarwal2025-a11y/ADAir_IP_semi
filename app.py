import argparse
import json
import os

import gradio as gr
import numpy as np

from utils.restorer import ImageRestorer


def build_app(restorer: ImageRestorer) -> gr.Blocks:
    def restore(npy_file, reference_file):
        if npy_file is None:
            return None, None, "Upload a NoisyLR .npy file to restore.", "{}", None
        try:
            input_path = npy_file if isinstance(npy_file, str) else npy_file.name
            reference_path = None
            if reference_file is not None:
                reference_path = reference_file if isinstance(reference_file, str) else reference_file.name

            input_view, restored_view, restored_array, status, params = restorer.restore_npy(
                input_path,
                reference_path,
            )

            os.makedirs("results", exist_ok=True)
            output_path = "results/restored_output.npy"
            npy_payload = restored_array.astype("float32")
            np.save(output_path, npy_payload)
            return input_view, restored_view, status, json.dumps(params, indent=2), output_path
        except Exception as exc:
            return None, None, f"Error: {exc}", "{}", None

    with gr.Blocks(title="AdaIR Restoration") as demo:
        gr.Markdown(
            """
            # AdaIR Wafer Image Restoration
            Upload a noisy low-resolution `.npy` array. Add a ground-truth `.npy` array when you want PSNR, SSIM, and LPIPS.
            """
        )

        with gr.Row():
            with gr.Column():
                input_file = gr.File(
                    label="NoisyLR input (.npy)",
                    file_types=[".npy"],
                    type="filepath",
                )
                reference_file = gr.File(
                    label="Ground truth / reference (.npy, optional)",
                    file_types=[".npy"],
                    type="filepath",
                )
                restore_btn = gr.Button("Restore", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Restored preview", type="numpy")
                download_btn = gr.DownloadButton(
                    label="Download restored .npy",
                    value=None,
                )

        with gr.Row():
            preview_input = gr.Image(label="Input preview", type="numpy")

        status = gr.Textbox(label="Status", interactive=False)
        params_box = gr.Code(label="Parameters and metrics", language="json")

        def run(npy_file, reference):
            return restore(npy_file, reference)

        restore_btn.click(
            fn=run,
            inputs=[input_file, reference_file],
            outputs=[preview_input, output_image, status, params_box, download_btn],
        )
        input_file.change(
            fn=run,
            inputs=[input_file, reference_file],
            outputs=[preview_input, output_image, status, params_box, download_btn],
        )
        reference_file.change(
            fn=run,
            inputs=[input_file, reference_file],
            outputs=[preview_input, output_image, status, params_box, download_btn],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    restorer = ImageRestorer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    demo = build_app(restorer)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
