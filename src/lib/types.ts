// ============================================================
// TypeScript 类型定义 - 对应后端 Python 模型
// ============================================================

// 文档与分块
export interface DocumentMetadata {
  doc_id: string;
  filename: string;
  file_type: string;
  title: string;
  source: string;
  created_at: number;
  chunk_count: number;
}

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  content: string;
  index: number;
  metadata: Record<string, unknown>;
  token_count: number;
  parent_chunk_id: string | null;
}

// 检索
export type MatchType = "vector" | "keyword" | "graph";

export interface SearchResult {
  chunk: Chunk;
  score: number;
  match_type: MatchType;
  highlighted_content: string;
}

// Agent
export interface ToolCall {
  call_id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResult {
  call_id: string;
  name: string;
  output: string;
  data: Record<string, unknown> | null;
  error: string | null;
  is_error: boolean;
}

export interface AgentStep {
  step_index: number;
  thought: string;
  tool_calls: ToolCall[];
  tool_results: ToolResult[];
  observation: string;
}

export type AgentState = "thinking" | "acting" | "observing" | "completed" | "error";

// Wiki
export type WikiPageType = "index" | "summary" | "entity" | "concept" | "synthesis";

export interface WikiPage {
  slug: string;
  title: string;
  page_type: WikiPageType;
  content: string;
  source_doc_ids: string[];
  source_chunk_ids: string[];
  out_links: string[];
  status: string;
  created_at: number;
  updated_at: number;
}

export interface WikiPageSummary {
  slug: string;
  title: string;
  type: WikiPageType;
  status: string;
  out_links: string[];
  updated_at: number;
}

// 知识图谱
export interface Entity {
  entity_id: string;
  title: string;
  description: string;
  entity_type: string;
  frequency: number;
  source_chunk_ids: string[];
}

export interface Relationship {
  relation_id: string;
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  description: string;
  weight: number;
  source_chunk_ids: string[];
}

export interface KnowledgeGraph {
  entities: Record<string, Entity>;
  relationships: Relationship[];
}

// API 请求/响应
export type ChatMode = "rag" | "agent" | "wiki";

export interface ChatRequest {
  query: string;
  mode: ChatMode;
  knowledge_base_ids?: string[];
  document_ids?: string[];
  conversation_id?: string;
  stream?: boolean;
  similarity_threshold_override?: number;
  max_context_turns?: number;
}

export interface ChatResponse {
  answer: string;
  sources: SearchResult[];
  agent_steps: AgentStep[];
  conversation_id: string;
  recommended_questions: string[];
}

export interface UploadResponse {
  doc_id: string;
  filename: string;
  chunk_count: number;
  message: string;
}

export interface WikiGenerateRequest {
  knowledge_base_id?: string;
  document_ids?: string[];
  granularity: "focused" | "standard" | "exhaustive";
}

export interface WikiGenerateResponse {
  task_id: string;
  status: string;
  pages_generated: number;
  message: string;
}

// 系统
export interface SystemStatus {
  status: string;
  documents: number;
  total_chunks: number;
  wiki_pages: number;
  graph_entities: number;
  graph_relationships: number;
  sessions: number;
}

export interface AgentTool {
  name: string;
  description: string;
}

export interface PromptTemplate {
  name: string;
  template: string;
  is_overridden: boolean;
  default_template?: string;
}

// 设置
export interface LLMConfig {
  api_key: string;
  base_url: string;
  chat_model: string;
  embedding_model: string;
  embedding_dim: number;
  max_tokens: number;
  temperature: number;
  timeout: number;
}

export interface RetrieverConfig {
  vector_top_k: number;
  keyword_top_k: number;
  rerank_top_k: number;
  similarity_threshold: number;
  mmr_lambda: number;
  hybrid_alpha: number;
}

export interface ChunkerConfig {
  chunk_size: number;
  chunk_overlap: number;
  separator: string;
  hierarchical: boolean;
  chunk_size_parent: number;
}

export interface AgentConfig {
  max_iterations: number;
  max_context_tokens: number;
  parallel_tool_calls: boolean;
  thinking_enabled: boolean;
  max_tool_output_size: number;
}

export interface MCPServerConfig {
  name: string;
  url: string;
  transport: "sse" | "http";
}

// 聊天消息（前端用）
export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: SearchResult[];
  agentSteps?: AgentStep[];
  recommendedQuestions?: string[];
  isStreaming?: boolean;
}
