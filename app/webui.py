import json
import gradio as gr

from summarizer import load_clean_article, setup_summarization_chain, clamp_text
from yt_summarizer import check_link, summarize_video


def _to_text(x):
    """把 LLM 返回统一转成字符串，兼容 AIMessage / dict / str。"""
    if hasattr(x, "content"):
        return x.content
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)


def summarize(url: str):
    # 点击时先显示“加载中”提示
    print("loading...", url)
    yield "⏳ 正在分析内容，请稍候..."

    try:
        if not url:
            yield "❌ 请输入 URL"
            return

        if check_link(url):
            result = summarize_video(url)
            text = _to_text(result)
        else:
            article = load_clean_article(url)
            content_text = clamp_text(f"{article.get('title', '')}\n\n{article.get('text', '')}")
            llm_chain = setup_summarization_chain()
            resp = llm_chain.invoke({"content": content_text})
            text = _to_text(resp)

        yield text
    except Exception as e:
        yield f"❌ 出错：{e}"


with gr.Blocks() as demo:
    gr.Markdown(
        "# Cobanov Web and Video Summarizer\nEasily summarize any web page or YouTube video with a single click.")

    with gr.Row():
        with gr.Column():
            url = gr.Text(label="URL", placeholder="Enter URL here")
            btn_generate = gr.Button("Generate")
            summary = gr.Markdown(label="Summary")

    gr.Examples(
        [
            "https://finance.sina.com.cn/meeting/2025-08-09/doc-infkkqsz7933277.shtml",
            "https://bawolf.substack.com/p/embeddings-are-a-good-starting-point",
            "https://www.youtube.com/watch?v=4pOpQwiUVXc",
        ],
        inputs=[url],
    )

    gr.Markdown(
        "```\nModel: gemma3:4b\nAuthor: YueYong\nContact: yueyong1030@outlook.com\nRepo: https://github.com/YueYongDev/easy-web-summarizer\n```")

    # 关键：outputs 第2个是按钮，用 gr.update 控制
    btn_generate.click(
        summarize,
        inputs=[url],
        outputs=[summary],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
