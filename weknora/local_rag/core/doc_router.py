"""
文档路由器 - 根据用户查询关键词匹配相关文档

在检索前增加一个「文档路由」步骤：
1. 从用户问题中提取关键词
2. 与文档名/标题/元数据进行匹配
3. 返回匹配的 doc_id 列表

这样当加载了很多文档时，检索只在相关文档中进行，避免噪声。
如果没有任何文档匹配，则回退到全量检索。
"""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DocRouter:
    """
    文档路由器

    核心功能：
    - 基于关键词的文档名/标题匹配（精确 > 子串 > 分词）
    - 基于元数据关键词的匹配（文档解析时提取的 section titles 等）
    - 支持同义词/别名映射（用户自定义）
    - 路由结果缓存（相同查询不重复计算）

    匹配策略（三重匹配，由严格到宽松）：
    1. 精确匹配：关键词 == 文档名（去掉扩展名）
    2. 子串匹配：关键词 in 文档名 或 文档名 in 关键词
    3. 分词匹配：对查询和文档名分词后，计算交集比例

    路由决策：
    - 匹配到文档 → 只在匹配的文档中检索
    - 没有匹配 → 全量检索（所有文档）
    """

    def __init__(self):
        # 文档注册表: doc_id -> DocRoutingInfo
        self._docs: Dict[str, DocRoutingInfo] = {}

        # 同义词映射: 别名 -> 标准名（可由用户自定义）
        # 例如: {"设备A": "XX型设备操作手册", "安全": "安全操作规程"}
        self._aliases: Dict[str, str] = {}

        # 路由结果缓存: query_hash -> doc_ids
        self._cache: Dict[str, List[str]] = {}

    def register_document(
        self,
        doc_id: str,
        filename: str = "",
        title: str = "",
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        注册一个文档到路由器

        Args:
            doc_id: 文档 ID
            filename: 文件名（含扩展名）
            title: 文档标题
            keywords: 文档关键词列表
            metadata: 文档元数据（如 section titles, tags 等）
        """
        info = DocRoutingInfo(
            doc_id=doc_id,
            filename=filename,
            title=title,
            keywords=keywords or [],
            metadata=metadata or {},
        )
        self._docs[doc_id] = info
        # 注册后清缓存
        self._cache.clear()
        logger.debug(f"文档路由器注册: {doc_id} -> {filename or title}")

    def unregister_document(self, doc_id: str):
        """移除文档注册"""
        if doc_id in self._docs:
            del self._docs[doc_id]
            self._cache.clear()

    def set_aliases(self, aliases: Dict[str, str]):
        """
        设置同义词/别名映射

        Args:
            aliases: {别名: 标准名} 映射

        示例:
            router.set_aliases({
                "设备A": "XX型设备操作手册",
                "安全规程": "安全操作规程",
                "SOP": "标准操作流程",
            })
        """
        self._aliases.update(aliases)
        self._cache.clear()

    def route(self, query: str, min_score: float = 0.4) -> List[str]:
        """
        对用户查询进行文档路由

        Args:
            query: 用户查询文本
            min_score: 最低匹配分数阈值（0~1），低于此值视为不匹配

        Returns:
            匹配的 doc_id 列表。如果为空，表示无匹配，应回退到全量检索。
        """
        if not self._docs:
            return []

        # 检查缓存
        cache_key = f"{query}::{min_score}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 提取查询关键词
        query_tokens = self._tokenize(query)
        query_lower = query.lower()

        # 对每个文档计算匹配分数
        scored_docs: List[Tuple[str, float]] = []

        for doc_id, info in self._docs.items():
            score = self._compute_match_score(query_lower, query_tokens, info)
            if score >= min_score:
                scored_docs.append((doc_id, score))

        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 提取 doc_id
        result = [doc_id for doc_id, _ in scored_docs]

        #  todo
        must_matched = ["19011156-SC_A20《H5U&Easy系列可编程逻辑控制器指令手册》", "19011157-SC_A19《H5U&Easy系列可编程逻辑控制器编程手册》"]
        must_matched = [m.lower() for m in must_matched]
        for doc_id, info in self._docs.items():
            if info.display_name.lower() in must_matched:
                result.append(doc_id)

        # 缓存
        self._cache[cache_key] = result

        if result:
            matched_names = [self._docs[did].display_name for did in result[:5]]
            logger.info(
                f"文档路由: 查询='{query[:50]}' → 匹配 {len(result)}/{len(self._docs)} 个文档: "
                f"{matched_names}"
            )
        else:
            logger.info(
                f"文档路由: 查询='{query[:50]}' → 无匹配，将使用全量检索"
            )

        return result

    def route_with_fallback(self, query: str, all_doc_ids: List[str], min_score: float = 0.3) -> List[str]:
        """
        文档路由（带回退）

        如果路由结果为空，回退到全量文档列表。

        Args:
            query: 用户查询
            all_doc_ids: 所有文档 ID 列表
            min_score: 最低匹配分数阈值

        Returns:
            匹配的 doc_id 列表，无匹配时返回 all_doc_ids
        """
        routed = self.route(query, min_score=min_score)
        if routed:
            return routed
        return all_doc_ids

    def _compute_match_score(
        self, query_lower: str, query_tokens: List[str], info: "DocRoutingInfo"
    ) -> float:
        """
        计算查询与文档的匹配分数

        匹配策略：
        1. 精确匹配（权重最高 1.0）：查询关键词 == 文档名（去掉扩展名）
        2. 子串匹配（权重 0.7）：关键词 in 文档名 或 文档名 in 关键词
        3. 分词匹配（权重按交集比例）：分词后计算交集
        4. 关键词匹配（权重 0.5）：文档关键词在查询中出现
        5. 别名匹配（权重 0.8）：同义词映射命中
        6. 元数据匹配（权重 0.4）：section title / tag 在查询中出现

        取所有策略中的最高分
        """
        max_score = 0.0

        # 文档名（去掉扩展名，转小写）
        doc_name = self._strip_extension(info.filename).lower()
        doc_title = info.title.lower()
        doc_display = info.display_name.lower()

        # 1. 精确匹配
        if query_lower == doc_name or query_lower == doc_title or query_lower == doc_display:
            return 1.0

        # 对查询中的每个 token 进行匹配
        for token in query_tokens:
            token_lower = token.lower()

            # 精确匹配 token
            if token_lower == doc_name or token_lower == doc_title:
                return 1.0

            # 2. 子串匹配
            if len(token_lower) >= 2:  # 至少2个字符才做子串匹配
                if token_lower in doc_name or token_lower in doc_title or token_lower in doc_display:
                    max_score = max(max_score, 0.7)
                # 反向：文档名包含在查询 token 中
                if doc_name and (doc_name in query_lower or doc_title in query_lower):
                    max_score = max(max_score, 0.7)

        # 3. 分词匹配 - 对文档名也分词，计算交集
        doc_tokens = self._tokenize(info.display_name)
        if query_tokens and doc_tokens:
            # 转小写集合
            query_set = set(t.lower() for t in query_tokens if len(t) >= 2)
            doc_set = set(t.lower() for t in doc_tokens if len(t) >= 2)

            if query_set and doc_set:
                intersection = query_set & doc_set
                if intersection:
                    # Jaccard 相似度的变体：交集 / 文档词数
                    ratio = len(intersection) / len(doc_set)
                    max_score = max(max_score, min(ratio * 1.2, 0.9))

        # 4. 关键词匹配
        for kw in info.keywords:
            kw_lower = kw.lower()
            if kw_lower in query_lower:
                max_score = max(max_score, 0.5)
                break
            # 分词匹配
            kw_tokens = set(t.lower() for t in self._tokenize(kw) if len(t) >= 2)
            query_set = set(t.lower() for t in query_tokens if len(t) >= 2)
            if kw_tokens and query_set and kw_tokens & query_set:
                max_score = max(max_score, 0.4)

        # 5. 别名匹配
        for alias, standard in self._aliases.items():
            alias_lower = alias.lower()
            standard_lower = standard.lower()
            # 查询中出现了别名
            if alias_lower in query_lower:
                # 检查标准名是否与此文档匹配
                if standard_lower in doc_name or standard_lower in doc_title or standard_lower in doc_display:
                    max_score = max(max_score, 0.8)
                    break
                # 标准名分词匹配
                std_tokens = set(t.lower() for t in self._tokenize(standard) if len(t) >= 2)
                doc_tokens_set = set(t.lower() for t in doc_tokens if len(t) >= 2)
                if std_tokens and doc_tokens_set and std_tokens & doc_tokens_set:
                    max_score = max(max_score, 0.7)
                    break

        # 6. 元数据匹配（section titles, tags 等）
        section_titles = info.metadata.get("section_titles", [])
        tags = info.metadata.get("tags", [])
        meta_texts = [t.lower() for t in section_titles + tags]

        for meta_text in meta_texts:
            if meta_text in query_lower:
                max_score = max(max_score, 0.4)
                break
            # 分词匹配
            meta_tokens = set(t.lower() for t in self._tokenize(meta_text) if len(t) >= 2)
            query_set = set(t.lower() for t in query_tokens if len(t) >= 2)
            if meta_tokens and query_set and meta_tokens & query_set:
                max_score = max(max_score, 0.3)

        return max_score

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        中文优化分词

        中文：尝试 jieba 分词，否则按 bigram
        英文：按空格分词
        """
        tokens: List[str] = []

        # 中文分词
        chinese_segments = re.findall(r'[\u4e00-\u9fff]+', text)
        try:
            import jieba
            for seg in chinese_segments:
                tokens.extend(jieba.lcut(seg))
        except ImportError:
            for seg in chinese_segments:
                tokens.extend(list(seg))
                for i in range(len(seg) - 1):
                    tokens.append(seg[i:i + 2])

        # 英文分词
        tokens.extend(re.findall(r'[a-zA-Z0-9]+', text.lower()))

        # 数字
        tokens.extend(re.findall(r'\d+', text))

        return [t for t in tokens if t.strip()]

    @staticmethod
    def _strip_extension(filename: str) -> str:
        """去掉文件扩展名"""
        if '.' in filename:
            return filename.rsplit('.', 1)[0]
        return filename

    def get_routing_info(self) -> List[Dict]:
        """获取路由信息（调试用）"""
        return [
            {
                "doc_id": info.doc_id,
                "display_name": info.display_name,
                "keywords": info.keywords,
                "aliases": [
                    alias for alias, std in self._aliases.items()
                    if std.lower() in info.display_name.lower()
                ],
            }
            for info in self._docs.values()
        ]


class DocRoutingInfo:
    """文档路由信息"""

    def __init__(
        self,
        doc_id: str,
        filename: str = "",
        title: str = "",
        keywords: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.title = title
        self.keywords = keywords or []
        self.metadata = metadata or {}

    @property
    def display_name(self) -> str:
        """显示名称：优先 title，其次 filename（去掉扩展名）"""
        if self.title:
            return self.title
        if self.filename:
            return DocRouter._strip_extension(self.filename)
        return self.doc_id
