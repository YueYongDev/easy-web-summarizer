import argparse
import os
import sys

# Updated imports to use new LangChain modules
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.document_loaders import WebBaseLoader
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


def load_document(url):
    """Load document from the specified URL."""
    # Set USER_AGENT to avoid warnings
    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    loader = WebBaseLoader(url)
    return loader.load()


def setup_summarization_chain():
    """Setup the summarization chain with a prompt template and ChatOllama."""
    prompt_template = PromptTemplate(
        template="""作为一名专业的中文摘要员，请创建所提供文本的详细而全面的中文摘要，无论是文章、帖子、对话还是段落，同时遵循以下准则：
            1. 制作一个详细、透彻、深入且复杂的摘要，同时保持清晰度。

            2. 融入主要观点和基本信息，消除多余的语言，专注于关键方面。

            3. 严格依赖提供的文本，不包括外部信息。

            4. 以段落形式格式化摘要，便于理解。

        通过遵循这个优化的提示，您将生成一个有效的摘要，以清晰、详细且对读者友好的方式封装给定文本的精髓。以markdown文件格式优化输出。

        "{text}"

        详细中文摘要:""",
        input_variables=["text"],
    )

    # Using the updated ChatOllama class
    llm = ChatOllama(model="gemma3:4b", base_url="http://127.0.0.1:11434")
    # Using RunnableSequence instead of deprecated LLMChain
    llm_chain = prompt_template | llm
    return llm_chain


def main():
    args = setup_argparse()
    docs = load_document(args.url)

    llm_chain = setup_summarization_chain()
    # Using invoke instead of deprecated run method
    result = llm_chain.invoke({"text": docs[0].page_content})
    # Print the result to see the output with proper encoding
    # Ensure proper encoding for Chinese characters
    if isinstance(result.content, str):
        print(result.content)
    else:
        print(result.content.decode('utf-8'))


if __name__ == "__main__":
    # Ensure UTF-8 encoding for proper Chinese character display
    sys.stdout.reconfigure(encoding='utf-8')
    main()