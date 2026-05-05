"""
推荐问题生成器 - 基于检索结果生成后续推荐问题
借鉴 WeKnora 的推荐问题生成设计
"""
import logging
from typing import List

from openai import AsyncOpenAI

from config.settings import AppConfig
from models.schemas import SearchResult

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    推荐问题生成器

    借鉴 WeKnora 的推荐问题设计：
    - 基于检索结果生成 3-5 个后续推荐问题
    - 问题自包含（不使用代词）
    - 问题可从知识库中找到答案
    - 语言感知（与文档语言一致）
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            timeout=config.llm.timeout,
        )

    async def generate(
        self,
        query: str,
        search_results: List[SearchResult],
        num_questions: int = 4,
    ) -> List[str]:
        """
        生成推荐后续问题

        Args:
            query: 用户原始查询
            search_results: 检索结果
            num_questions: 生成问题数量（3-5）

        Returns:
            推荐问题列表
        """
        if not search_results:
            return []

        # 构建上下文摘要
        context_parts = []
        for r in search_results[:5]:
            preview = r.chunk.content[:200].replace("\n", " ")
            context_parts.append(preview)
        context_summary = "\n".join(context_parts)

        # 检测语言
        language = self._detect_language(query, search_results)

        # 尝试使用模板管理器
        prompt = self._build_prompt(query, context_summary, language, num_questions)

        try:
            response = await self._client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )

            result_text = response.choices[0].message.content.strip()
            questions = self._parse_questions(result_text)

            return questions[:num_questions]

        except Exception as e:
            logger.warning(f"生成推荐问题失败: {e}")
            return []

    def _build_prompt(
        self, query: str, context: str, language: str, num_questions: int
    ) -> str:
        """构建生成提示词"""
        # 尝试使用模板管理器
        try:
            from agent.prompts import _get_template_manager
            manager = _get_template_manager()
            prompt = manager.get_prompt(
                "generate_questions",
                query=query,
                context=context[:1000],
                language=language,
            )
            if prompt:
                return prompt
        except Exception:
            pass

        # 后备提示词
        return f"""根据以下检索结果和用户的问题，生成 {num_questions} 个推荐的后续问题。

用户问题：{query}
检索结果摘要：{context[:1000]}

要求：
1. 问题应自包含，不使用代词
2. 问题应能从知识库中找到答案
3. 使用与文档相同的语言（{language}）
4. 覆盖不同角度和深度

按以下格式输出，每行一个问题："""

    @staticmethod
    def _detect_language(query: str, results: List[SearchResult]) -> str:
        """检测查询和文档的语言"""
        import re
        # 检查是否包含中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        if chinese_chars > len(query) * 0.2:
            return "中文"

        # 检查检索结果
        if results:
            sample = results[0].chunk.content[:200]
            chinese_in_sample = len(re.findall(r'[\u4e00-\u9fff]', sample))
            if chinese_in_sample > len(sample) * 0.2:
                return "中文"

        return "English"

    @staticmethod
    def _parse_questions(text: str) -> List[str]:
        """解析生成的问题文本"""
        questions = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除编号前缀（如 "1. ", "1) ", "- "）
            import re
            line = re.sub(r'^[\d]+[\.\)、]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line and len(line) > 5:  # 过滤掉太短的行
                questions.append(line)
        return questions
