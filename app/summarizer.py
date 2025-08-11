import argparse
import json
import os
import re
import sys
from typing import List

# 添加 traifatura 用于更好的网页内容提取
import trafilatura
# Updated imports to use new LangChain modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
# Fix import for ChatOllama
from langchain_ollama import ChatOllama


def setup_argparse():
    """Setup argparse to parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize a document from a given URL."
    )
    parser.add_argument(
        "-u", "--url", required=True, help="URL of the document to summarize"
    )
    return parser.parse_args()


def load_clean_article(url: str) -> dict:
    """
    优先用 trafilatura 抽正文；失败则回落到 SeleniumURLLoader。
    对于 juejin.im 等域名的文章，直接使用 SeleniumURLLoader。
    返回: {title, date, text, url}
    """
    # 特殊处理 域名
    special_domains = ["juejin.cn", "163.com", "guokr.com", "baidu.com", "smzdm.com", "nmc.cn", "52pojie.cn",
                       "toutiao.com", "sspai.com", "sina.com.cn", "hupu.com", "51cto.com", "ithome.com",
                       "news.qq.com", "nodeseek.com", "thepaper.cn", "hellogithub.com", "miyoushe.com"]  # 维护特殊域名列表
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


def setup_summarization_chain():
    """Setup the summarization chain with a prompt template and ChatOllama."""
    # 避免使用 { }，以免被模板引擎当成变量
    system = (
        "你是一个只输出 JSON 的助手。绝不输出任何解释、前后缀、Markdown 代码块或反引号。"
        "输出必须是合法 JSON 对象（最外层一对花括号）。"
    )

    user_template = """
基于给定中文文章正文，生成结构化摘要与标签，严格返回 JSON，字段与约束如下：
- summary: 120~220 字的中文摘要；不出现换行；只依据给定正文；不加入外部信息与日期推测。
- tags: 3~6 个中文标签，名词或短语，按重要性降序；不得包含"标签"二字；不含标点。

请只返回 JSON。禁止输出 Markdown、禁止 ```、禁止说明文字。

正文（含标题）：
{content}
"""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_template),  # 这里只保留一个 {content} 变量
    ])

    model_name = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0,
        format="json",
        num_ctx=8192,
    )
    return chat_prompt | llm


def parse_json_safely(s: str):
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
    return obj


def main():
    args = setup_argparse()

    try:
        article = load_clean_article(args.url)
        content_text = clamp_text(f"{article['title']}\n\n{article['text']}")

        chain = setup_summarization_chain()
        result = chain.invoke({"content": content_text})

        # 解析结果
        raw = result.content
        try:
            parsed = parse_json_safely(raw)
            summary = parsed.get("summary", "")
            tags = parsed.get("tags", [])
        except (ValueError, json.JSONDecodeError):
            # 兜底：给出标题+占位
            summary = (article["title"] or "摘要生成失败").strip()[:100]
            if not summary:
                summary = "摘要生成失败"
            tags = ["生成失败", "回退"]

        # 输出结果
        print(f"标题: {article['title']}")
        print(f"摘要: {summary}")
        print(f"标签: {', '.join(tags)}")

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure UTF-8 encoding for proper Chinese character display
    sys.stdout.reconfigure(encoding='utf-8')
    main()
