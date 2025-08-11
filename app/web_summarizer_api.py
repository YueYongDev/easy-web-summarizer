# web_summarizer_api.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from typing import List, Optional

# 抽正文
import trafilatura
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
# LLM & Prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

import logging

# ========= 日志配置 =========
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG 会输出全部日志
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# 确保终端/日志编码
os.environ["PYTHONIOENCODING"] = "utf-8"

app = FastAPI(
    title="Web Summarizer API",
    description="API for summarizing web content using LLM",
    version="1.1.0",
)


# =========================
# --------- I/O -----------
# =========================

class URLRequest(BaseModel):
    url: str


class SummaryResponse(BaseModel):
    summary: str
    tags: List[str]


# 输出严格校验（服务端自查）
class SummOut(BaseModel):
    summary: str = Field(min_length=40, max_length=500)
    tags: List[str] = Field(min_items=3, max_items=8)


# =========================
# ---- 抽取/裁剪工具 -------
# =========================

def load_clean_article(url: str) -> dict:
    """
    优先用 trafilatura 抽正文；失败则回落到 SeleniumURLLoader。
    对于 juejin.im 等域名的文章，直接使用 SeleniumURLLoader。
    返回: {title, date, text, url}
    """
    # 特殊处理 域名
    special_domains = ["juejin.cn", "163.com"]  # 维护特殊域名列表
    if any(domain in url for domain in special_domains):
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        docs = SeleniumURLLoader([url]).load()
        if not docs:
            raise ValueError("无法加载页面内容")
        page = docs[0]
        title = page.metadata.get("title") or ""
        text = page.page_content or ""
        if not text.strip():
            raise ValueError("页面内容为空")
        return {"title": title.strip(), "date": "", "text": text.strip(), "url": url}

    # 1) trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            jtxt = trafilatura.extract(
                downloaded, output_format="json", with_metadata=True,
                favor_recall=False, include_comments=False, include_images=False
            )
            if jtxt:
                jd = json.loads(jtxt)
                title = (jd.get("title") or "").strip()
                text = (jd.get("text") or "").strip()
                if text:
                    return {"title": title, "date": jd.get("date") or "", "text": text, "url": url}
    except Exception:
        pass

    # 2) 回退：LangChain WebBaseLoader
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    docs = WebBaseLoader(url).load()
    if not docs:
        raise ValueError("无法加载页面内容")
    page = docs[0]
    title = page.metadata.get("title") or ""
    text = page.page_content or ""
    if not text.strip():
        raise ValueError("页面内容为空")
    return {"title": title.strip(), "date": "", "text": text.strip(), "url": url}


def clamp_text(t: str, max_chars: int = 4000) -> str:
    """长文裁剪：首2000 + 尾1000，中间可留占位。"""
    t = re.sub(r"[ \t]+\n", "\n", t)  # 行尾空白
    if len(t) <= max_chars:
        return t
    head = t[:2000]
    tail = t[-1000:]
    return head + "\n……\n" + tail


# =========================
# ---- Prompt & LLM --------
# =========================

def setup_summarization_chain(model_name: Optional[str] = None):
    # 避免使用 { }，以免被模板引擎当成变量
    system = (
        "你是一个只输出 JSON 的助手。绝不输出任何解释、前后缀、Markdown 代码块或反引号。"
        "输出必须是合法 JSON 对象（最外层一对花括号）。"
    )

    user_template = """
基于给定中文文章正文，生成结构化摘要与标签，严格返回 JSON，字段与约束如下：
- summary: 120~220 字的中文摘要；不出现换行；只依据给定正文；不加入外部信息与日期推测。
- tags: 3~6 个中文标签，名词或短语，按重要性降序；不得包含“标签”二字；不含标点。

请只返回 JSON。禁止输出 Markdown、禁止 ```、禁止说明文字。

正文（含标题）：
{content}
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_template),   # 这里只保留一个 {content} 变量
    ])

    model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:4b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        format="json",
        num_ctx=8192,
        reasoning=False
    )
    return chat_prompt | llm


# =========================
# ------ 解析与兜底 -------
# =========================

def parse_json_safely(s: str) -> SummOut:
    """
    稳健 JSON 解析：
    - 去掉 ```json 围栏
    - 提取首个 {...}
    - Pydantic 校验
    """
    s = s or ""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(),
               flags=re.IGNORECASE | re.MULTILINE)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        raise ValueError("未找到 JSON 对象")
    obj = json.loads(m.group(0))
    return SummOut(**obj)


# =========================
# --------- 路由 ----------
# =========================

@app.get("/")
async def root():
    return {"message": "Web Summarizer API is running"}


@app.post("/summarize", response_model=SummaryResponse)
async def summarize_url(request: URLRequest):
    try:
        logger.debug(f"收到请求 URL: {request.url}")

        article = load_clean_article(request.url)
        logger.debug(f"抽取正文成功，标题: {article['title']}, 正文字数: {len(article['text'])}")

        content_text = clamp_text(f"{article['title']}\n\n{article['text']}")
        logger.debug(f"裁剪后正文长度: {len(content_text)}")

        chain = setup_summarization_chain()
        result = chain.invoke({"content": content_text})

        raw = result.content
        logger.debug(f"LLM 原始输出: {raw!r}")

        try:
            parsed = parse_json_safely(raw)
            summary, tags = parsed.summary, parsed.tags
            logger.debug(f"JSON 解析成功, 摘要长度: {len(summary)}, 标签: {tags}")
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"JSON 解析失败: {e}")
            summary = (article["title"] or "摘要解析失败").strip()[:100]
            if not summary:
                summary = "摘要解析失败"
            tags = ["解析失败", "回退"]

        return SummaryResponse(summary=summary, tags=tags)

    except Exception as e:
        logger.exception(f"summarize_url 异常: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ---------- 入口 ---------
# =========================

def main():
    parser = argparse.ArgumentParser(description="Run the Web Summarizer API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")  # 兼容你的 curl
    args = parser.parse_args()

    # 设置标准输出编码
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+
    except Exception:
        pass

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
