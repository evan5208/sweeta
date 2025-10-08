import gradio as gr
import subprocess
import os

def remove_watermark(input_video, max_bbox_percent=10.0, output_format="MP4"):
    output_path = "output." + output_format.lower()
    cmd = f"python remwm.py {input_video} {output_path} --max-bbox-percent {max_bbox_percent} --force-format {output_format} --overwrite"
    subprocess.run(cmd, shell=True, check=True)
    if os.path.exists(output_path):
        return output_path
    else:
        raise Exception("去水印失败，请检查输入文件或参数。")

iface = gr.Interface(
    fn=remove_watermark,
    inputs=[
        gr.Video(label="上传 SORA 2 视频（MP4/AVI）"),
        gr.Slider(1, 100, value=10, label="水印检测敏感度（%）"),
        gr.Dropdown(["MP4", "AVI", "PNG", "WEBP", "JPG"], label="输出格式", value="MP4")
    ],
    outputs=gr.Video(label="去水印后的视频"),
    title="Sweeta 水印移除器",
    description="上传 SORA 2 视频，移除水印。仅限教育/研究用途，基于 IOPaint 和 LaMA 模型。"
)

if __name__ == "__main__":
    iface.launch()
