"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Send, Loader2, PanelLeftClose, PanelLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatSidebar } from "@/components/chat-sidebar";
import { ChatMessage } from "@/components/chat-message";
import { api } from "@/lib/api";
import type { ChatMessage as ChatMessageType, ChatMode, DocumentMetadata } from "@/lib/types";

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [mode, setMode] = useState<ChatMode>("rag");
  const [documents, setDocuments] = useState<DocumentMetadata[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // 检索设置
  const [similarityThreshold, setSimilarityThreshold] = useState(0.5);
  const [topK, setTopK] = useState(10);
  const [deepMode, setDeepMode] = useState(false);
  const [hierarchicalChunking, setHierarchicalChunking] = useState(false);
  const [conversationId, setConversationId] = useState<string | undefined>();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 加载文档列表
  const loadDocuments = useCallback(async () => {
    try {
      const res = await api.listDocuments();
      setDocuments(res.documents);
    } catch {
      // 静默处理
    }
  }, []);

  // 初始化加载文档（通过事件订阅模式避免 effect 内直接调用 setState）
  const loadDocumentsRef = useRef(false);
  useEffect(() => {
    if (!loadDocumentsRef.current) {
      loadDocumentsRef.current = true;
      const controller = new AbortController();
      api.listDocuments().then((res) => {
        if (!controller.signal.aborted) {
          setDocuments(res.documents);
        }
      }).catch(() => {});
      return () => controller.abort();
    }
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 监听推荐问题点击
  useEffect(() => {
    const handler = (e: CustomEvent) => {
      setInput(e.detail);
      textareaRef.current?.focus();
    };
    window.addEventListener("recommended-question-click", handler as EventListener);
    return () =>
      window.removeEventListener("recommended-question-click", handler as EventListener);
  }, []);

  const handleSend = useCallback(async () => {
    const query = input.trim();
    if (!query || isStreaming) return;

    setInput("");
    setIsStreaming(true);

    // 添加用户消息
    const userMsg: ChatMessageType = {
      id: Date.now().toString(),
      role: "user",
      content: query,
    };
    setMessages((prev) => [...prev, userMsg]);

    // 添加空的助手消息（用于流式填充）
    const assistantId = (Date.now() + 1).toString();
    const assistantMsg: ChatMessageType = {
      id: assistantId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };
    setMessages((prev) => [...prev, assistantMsg]);

    // 流式请求
    abortRef.current = api.chatStream(
      {
        query,
        mode,
        conversation_id: conversationId,
        stream: true,
        similarity_threshold_override: similarityThreshold,
      },
      // onChunk
      (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: m.content + data }
              : m
          )
        );
      },
      // onDone
      () => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m
          )
        );
        setIsStreaming(false);
      },
      // onError
      (error) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `错误: ${error}`, isStreaming: false }
              : m
          )
        );
        setIsStreaming(false);
      }
    );
  }, [input, isStreaming, mode, conversationId, similarityThreshold]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleClearHistory = useCallback(() => {
    if (isStreaming && abortRef.current) {
      abortRef.current.abort();
    }
    setMessages([]);
    setConversationId(undefined);
    setIsStreaming(false);
  }, [isStreaming]);

  // 欢迎消息
  const modeLabels: Record<ChatMode, string> = {
    rag: "RAG快速问答",
    agent: "Agent智能推理",
    wiki: "Wiki模式",
  };

  return (
    <div className="flex h-[calc(100vh-44px)]">
      {/* 左侧栏 */}
      {sidebarOpen && (
        <div className="w-72 border-r bg-muted/30 shrink-0">
          <ChatSidebar
            mode={mode}
            onModeChange={setMode}
            documents={documents}
            onDocumentsChange={loadDocuments}
            similarityThreshold={similarityThreshold}
            onSimilarityThresholdChange={setSimilarityThreshold}
            topK={topK}
            onTopKChange={setTopK}
            deepMode={deepMode}
            onDeepModeChange={setDeepMode}
            hierarchicalChunking={hierarchicalChunking}
            onHierarchicalChunkingChange={setHierarchicalChunking}
            onClearHistory={handleClearHistory}
          />
        </div>
      )}

      {/* 主聊天区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶部栏 */}
        <div className="flex items-center gap-2 px-4 py-2 border-b">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="h-4 w-4" />
            ) : (
              <PanelLeft className="h-4 w-4" />
            )}
          </Button>
          <span className="text-xs text-muted-foreground">
            当前模式: <span className="font-medium text-foreground">{modeLabels[mode]}</span>
          </span>
        </div>

        {/* 消息列表 */}
        <ScrollArea className="flex-1 px-4">
          <div className="max-w-3xl mx-auto py-6 space-y-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="h-16 w-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                  <svg className="h-8 w-8 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                </div>
                <h2 className="text-lg font-semibold mb-2">Local RAG 智能问答</h2>
                <p className="text-sm text-muted-foreground max-w-md">
                  上传文档后即可开始提问。支持 RAG 快速问答、Agent 多步推理和 Wiki 知识库查询。
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-8 max-w-lg">
                  {[
                    { label: "RAG快速问答", desc: "基于文档检索的快速回答", mode: "rag" as ChatMode },
                    { label: "Agent智能推理", desc: "多步推理与工具调用", mode: "agent" as ChatMode },
                    { label: "Wiki模式", desc: "查询Wiki知识库", mode: "wiki" as ChatMode },
                  ].map((item) => (
                    <button
                      key={item.mode}
                      className="p-3 rounded-lg border hover:border-primary/50 hover:bg-muted/50 transition-colors text-left"
                      onClick={() => setMode(item.mode)}
                    >
                      <p className="text-sm font-medium">{item.label}</p>
                      <p className="text-xs text-muted-foreground mt-1">{item.desc}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* 输入框 */}
        <div className="border-t px-4 py-3">
          <div className="max-w-3xl mx-auto flex gap-2 items-end">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`输入问题... (当前: ${modeLabels[mode]})`}
              className="min-h-[40px] max-h-[120px] resize-none"
              rows={1}
              disabled={isStreaming}
            />
            <Button
              size="sm"
              className="h-10 w-10 shrink-0"
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
