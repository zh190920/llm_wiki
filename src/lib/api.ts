/**
 * API 客户端模块 - 封装所有后端 API 调用
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, {
      ...options,
      headers: {
        ...options?.headers,
      },
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || `请求失败: ${res.status}`);
    }
    return res.json();
  }

  // ============================================================
  // 文档管理
  // ============================================================

  async uploadDocument(file: File): Promise<import("./types").UploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    const url = `${this.baseUrl}/api/documents/upload`;
    const res = await fetch(url, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || "上传失败");
    }
    return res.json();
  }

  async listDocuments(): Promise<{ documents: import("./types").DocumentMetadata[]; total: number }> {
    return this.request("/api/documents");
  }

  async deleteDocument(docId: string): Promise<{ message: string }> {
    return this.request(`/api/documents/${docId}`, { method: "DELETE" });
  }

  // ============================================================
  // 聊天
  // ============================================================

  async chat(request: import("./types").ChatRequest): Promise<import("./types").ChatResponse> {
    return this.request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  }

  /**
   * 流式聊天 - SSE
   * @param request 聊天请求
   * @param onChunk 每个数据块的回调
   * @param onDone 完成回调
   * @param onError 错误回调
   * @returns AbortController 用于取消请求
   */
  chatStream(
    request: import("./types").ChatRequest,
    onChunk: (data: string) => void,
    onDone: () => void,
    onError: (error: string) => void
  ): AbortController {
    const controller = new AbortController();
    const url = `${this.baseUrl}/api/chat/stream`;

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...request, stream: true }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          const error = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(error.detail || "流式请求失败");
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error("无法获取响应流");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
              const data = trimmed.slice(6);
              if (data === "[DONE]") {
                onDone();
                return;
              }
              if (data.startsWith("[ERROR]")) {
                onError(data.slice(8).trim());
                return;
              }
              onChunk(data);
            }
          }
        }
        onDone();
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          onError(err.message || "连接失败");
        }
      });

    return controller;
  }

  // ============================================================
  // Wiki
  // ============================================================

  async generateWiki(
    request: import("./types").WikiGenerateRequest
  ): Promise<import("./types").WikiGenerateResponse> {
    return this.request("/api/wiki/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  }

  async listWikiPages(
    pageType?: string,
    status?: string
  ): Promise<{ pages: import("./types").WikiPageSummary[]; total: number }> {
    const params = new URLSearchParams();
    if (pageType) params.set("page_type", pageType);
    if (status) params.set("status", status);
    const query = params.toString();
    return this.request(`/api/wiki/pages${query ? `?${query}` : ""}`);
  }

  async getWikiPage(slug: string): Promise<import("./types").WikiPage> {
    return this.request(`/api/wiki/pages/${slug}`);
  }

  async exportWiki(): Promise<{ message: string; path: string }> {
    return this.request("/api/wiki/export");
  }

  // ============================================================
  // 知识图谱
  // ============================================================

  async buildKnowledgeGraph(
    docIds?: string[]
  ): Promise<{
    entities: number;
    relationships: number;
    mermaid: string;
    graph_file: string;
    mermaid_file: string;
  }> {
    return this.request("/api/graph/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(docIds || []),
    });
  }

  async getGraphMermaid(): Promise<{ mermaid: string }> {
    return this.request("/api/graph/mermaid");
  }

  // ============================================================
  // 系统
  // ============================================================

  async getSystemStatus(): Promise<import("./types").SystemStatus> {
    return this.request("/api/system/status");
  }

  async listAgentTools(): Promise<{ tools: import("./types").AgentTool[] }> {
    return this.request("/api/system/tools");
  }

  async listPrompts(): Promise<{ templates: import("./types").PromptTemplate[] }> {
    return this.request("/api/system/prompts");
  }

  async updatePrompt(
    name: string,
    template: string
  ): Promise<{ message: string }> {
    return this.request(`/api/system/prompts/${name}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template }),
    });
  }

  async resetPrompt(name: string): Promise<{ message: string }> {
    return this.request(`/api/system/prompts/${name}`, {
      method: "DELETE",
    });
  }
}

// 单例导出
export const api = new ApiClient();
export { ApiClient };
