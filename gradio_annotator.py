import gradio as gr
from annotator.util import resize_image, HWC3


model_canny = None


def canny(img, res, l, h):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return [result]


model_hed = None


def hed(img, res):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return [result]


model_mlsd = None


def mlsd(img, res, thr_v, thr_d):
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector()
    result = model_mlsd(img, thr_v, thr_d)
    return [result]


model_midas = None


def midas(img, res, a):
    img = resize_image(HWC3(img), res)
    global model_midas
    if model_midas is None:
        from annotator.midas import MidasDetector
        model_midas = MidasDetector()
    results = model_midas(img, a)
    return results


model_openpose = None


def openpose(img, res, has_hand):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result, _ = model_openpose(img, has_hand)
    return [result]


model_uniformer = None


def uniformer(img, res):
    img = resize_image(HWC3(img), res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return [result]


block = gr.Blocks().queue()
with block:
    # --- CANNY EDGE ---
    with gr.Row():
        gr.Markdown("## Canny Edge")
    with gr.Row():
        with gr.Column():
            input_image_canny = gr.Image(sources=['upload'], type="numpy")
            low_threshold = gr.Slider(label="low_threshold", minimum=1, maximum=255, value=100, step=1)
            high_threshold = gr.Slider(label="high_threshold", minimum=1, maximum=255, value=200, step=1)
            resolution_canny = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button_canny = gr.Button(value="Run")
        with gr.Column():
            gallery_canny = gr.Gallery(label="Generated images", show_label=False, height="auto")
    run_button_canny.click(fn=canny, inputs=[input_image_canny, resolution_canny, low_threshold, high_threshold], outputs=[gallery_canny])

    # --- HED EDGE ---
    with gr.Row():
        gr.Markdown("## HED Edge")
    with gr.Row():
        with gr.Column():
            input_image_hed = gr.Image(sources=['upload'], type="numpy")
            resolution_hed = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button_hed = gr.Button(value="Run")
        with gr.Column():
            gallery_hed = gr.Gallery(label="Generated images", show_label=False, height="auto")
    run_button_hed.click(fn=hed, inputs=[input_image_hed, resolution_hed], outputs=[gallery_hed])
    # --- MLSD EDGE ---
    with gr.Row():
        gr.Markdown("## MLSD Edge")
    with gr.Row():
        with gr.Column():
            input_image_mlsd = gr.Image(sources=['upload'], type="numpy")
            value_threshold = gr.Slider(label="value_threshold", minimum=0.01, maximum=2.0, value=0.1, step=0.01)
            distance_threshold = gr.Slider(label="distance_threshold", minimum=0.01, maximum=20.0, value=0.1, step=0.01)
            resolution_mlsd = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button_mlsd = gr.Button(value="Run")
        with gr.Column():
            gallery_mlsd = gr.Gallery(label="Generated images", show_label=False, height="auto")
    run_button_mlsd.click(fn=mlsd, inputs=[input_image_mlsd, resolution_mlsd, value_threshold, distance_threshold], outputs=[gallery_mlsd])

    # --- MIDAS DEPTH ---
    with gr.Row():
        gr.Markdown("## MIDAS Depth and Normal")
    with gr.Row():
        with gr.Column():
            input_image_midas = gr.Image(sources=['upload'], type="numpy")
            alpha = gr.Slider(label="alpha", minimum=0.1, maximum=20.0, value=6.2, step=0.01)
            resolution_midas = gr.Slider(label="resolution", minimum=256, maximum=1024, value=384, step=64)
            run_button_midas = gr.Button(value="Run")
        with gr.Column():
            gallery_midas = gr.Gallery(label="Generated images", show_label=False, height="auto")
    run_button_midas.click(fn=midas, inputs=[input_image_midas, resolution_midas, alpha], outputs=[gallery_midas])

    # --- OPENPOSE ---
    with gr.Row():
        gr.Markdown("## Openpose")
    with gr.Row():
        with gr.Column():
            input_image_openpose = gr.Image(sources=['upload'], type="numpy")
            hand = gr.Checkbox(label='detect hand', value=False)
            resolution_openpose = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button_openpose = gr.Button(value="Run")
        with gr.Column():
            gallery_openpose = gr.Gallery(label="Generated images", show_label=False, height="auto")
    run_button_openpose.click(fn=openpose, inputs=[input_image_openpose, resolution_openpose, hand], outputs=[gallery_openpose])

    # --- UNIFORMER ---
    with gr.Row():
        gr.Markdown("## Uniformer Segmentation")
    with gr.Row():
        with gr.Column():
            input_image_uniformer = gr.Image(sources=['upload'], type="numpy")
            resolution_uniformer = gr.Slider(label="resolution", minimum=256, maximum=1024, value=512, step=64)
            run_button_uniformer = gr.Button(value="Run")
        with gr.Column():
            gallery_uniformer = gr.Gallery(label="Generated images", show_label=False, height="auto")
    run_button_uniformer.click(fn=uniformer, inputs=[input_image_uniformer, resolution_uniformer], outputs=[gallery_uniformer])


block.launch(server_name='0.0.0.0',share=True)