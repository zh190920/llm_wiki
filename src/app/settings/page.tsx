"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Settings,
  Cpu,
  Scissors,
  Search,
  Bot,
  Server,
  Activity,
  Plus,
  Trash2,
  Loader2,
  RefreshCw,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/lib/api";
import type { SystemStatus, MCPServerConfig } from "@/lib/types";
import { useToast } from "@/hooks/use-toast";

export default function SettingsPage() {
  const { toast } = useToast();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);

  // LLM 配置
  const [llmConfig, setLlmConfig] = useState({
    api_key: "",
    base_url: "https://api.openai.com/v1",
    chat_model: "gpt-4o-mini",
    embedding_model: "text-embedding-3-small",
    temperature: 0.3,
  });

  // 分块配置
  const [chunkerConfig, setChunkerConfig] = useState({
    chunk_size: 512,
    chunk_overlap: 64,
    hierarchical: false,
    chunk_size_parent: 2048,
  });

  // 检索配置
  const [retrieverConfig, setRetrieverConfig] = useState({
    similarity_threshold: 0.5,
    top_k: 10,
    rerank_top_k: 5,
    hybrid_alpha: 0.7,
  });

  // Agent 配置
  const [agentConfig, setAgentConfig] = useState({
    max_iterations: 10,
    max_context_tokens: 128000,
    parallel_tool_calls: true,
  });

  // MCP 服务器
  const [mcpServers, setMcpServers] = useState<MCPServerConfig[]>([]);

  const loadStatus = useCallback(async () => {
    setStatusLoading(true);
    try {
      const res = await api.getSystemStatus();
      setStatus(res);
    } catch (err) {
      toast({
        title: "加载状态失败",
        description: err instanceof Error ? err.message : "未知错误",
        variant: "destructive",
      });
    } finally {
      setStatusLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const handleSaveConfig = useCallback(
    (section: string) => {
      toast({
        title: "配置已保存",
        description: `${section} 配置已保存（需要重启后端生效）`,
      });
    },
    [toast]
  );

  const addMcpServer = useCallback(() => {
    setMcpServers((prev) => [
      ...prev,
      { name: "", url: "", transport: "sse" },
    ]);
  }, []);

  const removeMcpServer = useCallback((index: number) => {
    setMcpServers((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const updateMcpServer = useCallback(
    (index: number, field: keyof MCPServerConfig, value: string) => {
      setMcpServers((prev) =>
        prev.map((s, i) => (i === index ? { ...s, [field]: value } : s))
      );
    },
    []
  );

  return (
    <div className="h-[calc(100vh-44px)] flex flex-col">
      <div className="px-6 py-3 border-b flex items-center justify-between">
        <h1 className="text-sm font-semibold">系统设置</h1>
        <Button
          variant="outline"
          size="sm"
          className="h-7 text-xs gap-1"
          onClick={loadStatus}
          disabled={statusLoading}
        >
          <RefreshCw
            className={`h-3 w-3 ${statusLoading ? "animate-spin" : ""}`}
          />
          刷新状态
        </Button>
      </div>

      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="llm" className="h-full flex flex-col">
          <div className="px-6 pt-3">
            <TabsList className="w-full justify-start">
              <TabsTrigger value="llm" className="text-xs gap-1">
                <Cpu className="h-3 w-3" />
                LLM
              </TabsTrigger>
              <TabsTrigger value="chunker" className="text-xs gap-1">
                <Scissors className="h-3 w-3" />
                分块
              </TabsTrigger>
              <TabsTrigger value="retriever" className="text-xs gap-1">
                <Search className="h-3 w-3" />
                检索
              </TabsTrigger>
              <TabsTrigger value="agent" className="text-xs gap-1">
                <Bot className="h-3 w-3" />
                Agent
              </TabsTrigger>
              <TabsTrigger value="mcp" className="text-xs gap-1">
                <Server className="h-3 w-3" />
                MCP
              </TabsTrigger>
              <TabsTrigger value="status" className="text-xs gap-1">
                <Activity className="h-3 w-3" />
                状态
              </TabsTrigger>
            </TabsList>
          </div>

          <ScrollArea className="flex-1">
            <div className="p-6 max-w-3xl">
              {/* LLM 配置 */}
              <TabsContent value="llm" className="mt-0 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Cpu className="h-4 w-4" />
                      LLM 配置
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label className="text-xs">API Key</Label>
                      <Input
                        type="password"
                        value={llmConfig.api_key}
                        onChange={(e) =>
                          setLlmConfig((c) => ({ ...c, api_key: e.target.value }))
                        }
                        placeholder="sk-..."
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">Base URL</Label>
                      <Input
                        value={llmConfig.base_url}
                        onChange={(e) =>
                          setLlmConfig((c) => ({
                            ...c,
                            base_url: e.target.value,
                          }))
                        }
                        placeholder="https://api.openai.com/v1"
                        className="h-8 text-xs"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-xs">Chat Model</Label>
                        <Input
                          value={llmConfig.chat_model}
                          onChange={(e) =>
                            setLlmConfig((c) => ({
                              ...c,
                              chat_model: e.target.value,
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Embedding Model</Label>
                        <Input
                          value={llmConfig.embedding_model}
                          onChange={(e) =>
                            setLlmConfig((c) => ({
                              ...c,
                              embedding_model: e.target.value,
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Temperature</Label>
                        <span className="text-[10px] text-muted-foreground">
                          {llmConfig.temperature.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        value={[llmConfig.temperature]}
                        min={0}
                        max={2}
                        step={0.1}
                        onValueChange={([v]) =>
                          setLlmConfig((c) => ({ ...c, temperature: v }))
                        }
                      />
                    </div>
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => handleSaveConfig("LLM")}
                    >
                      保存配置
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* 分块配置 */}
              <TabsContent value="chunker" className="mt-0 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Scissors className="h-4 w-4" />
                      分块配置
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-xs">Chunk Size</Label>
                        <Input
                          type="number"
                          value={chunkerConfig.chunk_size}
                          onChange={(e) =>
                            setChunkerConfig((c) => ({
                              ...c,
                              chunk_size: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Chunk Overlap</Label>
                        <Input
                          type="number"
                          value={chunkerConfig.chunk_overlap}
                          onChange={(e) =>
                            setChunkerConfig((c) => ({
                              ...c,
                              chunk_overlap: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-xs">层级分块</Label>
                        <p className="text-[10px] text-muted-foreground">
                          启用父子块分层结构
                        </p>
                      </div>
                      <Switch
                        checked={chunkerConfig.hierarchical}
                        onCheckedChange={(v) =>
                          setChunkerConfig((c) => ({ ...c, hierarchical: v }))
                        }
                      />
                    </div>
                    {chunkerConfig.hierarchical && (
                      <div className="space-y-2">
                        <Label className="text-xs">Parent Chunk Size</Label>
                        <Input
                          type="number"
                          value={chunkerConfig.chunk_size_parent}
                          onChange={(e) =>
                            setChunkerConfig((c) => ({
                              ...c,
                              chunk_size_parent: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                    )}
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => handleSaveConfig("分块")}
                    >
                      保存配置
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* 检索配置 */}
              <TabsContent value="retriever" className="mt-0 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Search className="h-4 w-4" />
                      检索配置
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">相似度阈值</Label>
                        <span className="text-[10px] text-muted-foreground">
                          {retrieverConfig.similarity_threshold.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        value={[retrieverConfig.similarity_threshold]}
                        min={0}
                        max={1}
                        step={0.05}
                        onValueChange={([v]) =>
                          setRetrieverConfig((c) => ({
                            ...c,
                            similarity_threshold: v,
                          }))
                        }
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-xs">Top-K</Label>
                        <Input
                          type="number"
                          value={retrieverConfig.top_k}
                          onChange={(e) =>
                            setRetrieverConfig((c) => ({
                              ...c,
                              top_k: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">Rerank Top-K</Label>
                        <Input
                          type="number"
                          value={retrieverConfig.rerank_top_k}
                          onChange={(e) =>
                            setRetrieverConfig((c) => ({
                              ...c,
                              rerank_top_k: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs">Hybrid Alpha (向量检索权重)</Label>
                        <span className="text-[10px] text-muted-foreground">
                          {retrieverConfig.hybrid_alpha.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        value={[retrieverConfig.hybrid_alpha]}
                        min={0}
                        max={1}
                        step={0.05}
                        onValueChange={([v]) =>
                          setRetrieverConfig((c) => ({
                            ...c,
                            hybrid_alpha: v,
                          }))
                        }
                      />
                      <p className="text-[10px] text-muted-foreground">
                        0 = 纯关键词检索, 1 = 纯向量检索
                      </p>
                    </div>
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => handleSaveConfig("检索")}
                    >
                      保存配置
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Agent 配置 */}
              <TabsContent value="agent" className="mt-0 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Bot className="h-4 w-4" />
                      Agent 配置
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-xs">最大迭代次数</Label>
                        <Input
                          type="number"
                          value={agentConfig.max_iterations}
                          onChange={(e) =>
                            setAgentConfig((c) => ({
                              ...c,
                              max_iterations: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs">最大上下文 Token</Label>
                        <Input
                          type="number"
                          value={agentConfig.max_context_tokens}
                          onChange={(e) =>
                            setAgentConfig((c) => ({
                              ...c,
                              max_context_tokens: Number(e.target.value),
                            }))
                          }
                          className="h-8 text-xs"
                        />
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label className="text-xs">并行工具调用</Label>
                        <p className="text-[10px] text-muted-foreground">
                          允许同时调用多个工具
                        </p>
                      </div>
                      <Switch
                        checked={agentConfig.parallel_tool_calls}
                        onCheckedChange={(v) =>
                          setAgentConfig((c) => ({
                            ...c,
                            parallel_tool_calls: v,
                          }))
                        }
                      />
                    </div>
                    <Button
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => handleSaveConfig("Agent")}
                    >
                      保存配置
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* MCP 服务器 */}
              <TabsContent value="mcp" className="mt-0 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Server className="h-4 w-4" />
                      MCP 服务器配置
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {mcpServers.length === 0 ? (
                      <p className="text-xs text-muted-foreground text-center py-4">
                        暂无 MCP 服务器配置
                      </p>
                    ) : (
                      <div className="space-y-3">
                        {mcpServers.map((server, index) => (
                          <div
                            key={index}
                            className="flex items-end gap-2 p-3 border rounded-lg"
                          >
                            <div className="flex-1 space-y-2">
                              <div className="grid grid-cols-3 gap-2">
                                <div>
                                  <Label className="text-[10px]">名称</Label>
                                  <Input
                                    value={server.name}
                                    onChange={(e) =>
                                      updateMcpServer(index, "name", e.target.value)
                                    }
                                    className="h-7 text-xs"
                                    placeholder="服务器名称"
                                  />
                                </div>
                                <div>
                                  <Label className="text-[10px]">URL</Label>
                                  <Input
                                    value={server.url}
                                    onChange={(e) =>
                                      updateMcpServer(index, "url", e.target.value)
                                    }
                                    className="h-7 text-xs"
                                    placeholder="http://localhost:3001"
                                  />
                                </div>
                                <div>
                                  <Label className="text-[10px]">传输协议</Label>
                                  <Select
                                    value={server.transport}
                                    onValueChange={(v) =>
                                      updateMcpServer(index, "transport", v)
                                    }
                                  >
                                    <SelectTrigger className="h-7 text-xs">
                                      <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="sse">SSE</SelectItem>
                                      <SelectItem value="http">HTTP</SelectItem>
                                    </SelectContent>
                                  </Select>
                                </div>
                              </div>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 w-7 p-0 text-destructive"
                              onClick={() => removeMcpServer(index)}
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs gap-1"
                      onClick={addMcpServer}
                    >
                      <Plus className="h-3 w-3" />
                      添加 MCP 服务器
                    </Button>
                    {mcpServers.length > 0 && (
                      <Button
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => handleSaveConfig("MCP")}
                      >
                        保存配置
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* 系统状态 */}
              <TabsContent value="status" className="mt-0 space-y-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      系统状态
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {statusLoading ? (
                      <div className="flex items-center justify-center py-8">
                        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                      </div>
                    ) : status ? (
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {[
                          {
                            label: "运行状态",
                            value: status.status,
                            icon: status.status === "running" ? CheckCircle : XCircle,
                            color: status.status === "running" ? "text-emerald-500" : "text-destructive",
                          },
                          { label: "文档数", value: status.documents, icon: Settings },
                          { label: "分块数", value: status.total_chunks, icon: Scissors },
                          { label: "Wiki 页面", value: status.wiki_pages, icon: Search },
                          {
                            label: "图谱实体",
                            value: status.graph_entities,
                            icon: Activity,
                          },
                          {
                            label: "图谱关系",
                            value: status.graph_relationships,
                            icon: Activity,
                          },
                          { label: "活跃会话", value: status.sessions, icon: Cpu },
                        ].map((item) => (
                          <div
                            key={item.label}
                            className="p-3 border rounded-lg text-center"
                          >
                            <item.icon
                              className={`h-4 w-4 mx-auto mb-1.5 ${
                                "color" in item ? item.color : "text-muted-foreground"
                              }`}
                            />
                            <p className="text-lg font-semibold">{item.value}</p>
                            <p className="text-[10px] text-muted-foreground">
                              {item.label}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground text-center py-8">
                        无法获取系统状态，请检查后端是否运行
                      </p>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </div>
          </ScrollArea>
        </Tabs>
      </div>
    </div>
  );
}
