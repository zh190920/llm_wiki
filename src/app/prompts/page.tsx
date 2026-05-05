"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Code2,
  Save,
  RotateCcw,
  Check,
  AlertCircle,
  Loader2,
  Eye,
  Pencil,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { api } from "@/lib/api";
import type { PromptTemplate } from "@/lib/types";
import { useToast } from "@/hooks/use-toast";

const templateDescriptions: Record<string, string> = {
  rag_system: "RAG 快速问答系统提示词",
  agent_rag: "Agent RAG 模式系统提示词",
  agent_pure: "Agent 纯推理模式系统提示词",
  wiki_summary: "Wiki 摘要生成提示词",
  wiki_entity_extraction: "Wiki 实体提取提示词",
  wiki_page_modify: "Wiki 页面创建/更新提示词",
  wiki_deduplication: "Wiki 去重判断提示词",
  wiki_index: "Wiki 索引生成提示词",
  graph_entity_extraction: "知识图谱实体提取提示词",
  query_understanding: "查询理解与改写提示词",
  generate_questions: "推荐问题生成提示词",
};

export default function PromptsPage() {
  const { toast } = useToast();
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewValues, setPreviewValues] = useState<Record<string, string>>({});

  const selectedTemplate = templates.find((t) => t.name === selectedName);

  // 加载模板列表
  const loadTemplates = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listPrompts();
      setTemplates(res.templates || []);
      if (res.templates?.length > 0 && !selectedName) {
        setSelectedName(res.templates[0].name);
      }
    } catch (err) {
      toast({
        title: "加载失败",
        description: err instanceof Error ? err.message : "无法加载提示词模板",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast, selectedName]);

  useEffect(() => {
    loadTemplates();
  }, [loadTemplates]);

  // 选中模板时加载内容
  useEffect(() => {
    if (selectedTemplate) {
      setEditContent(selectedTemplate.template);
      // 提取占位符
      const placeholders = extractPlaceholders(selectedTemplate.template);
      const values: Record<string, string> = {};
      for (const p of placeholders) {
        values[p] = `[${p}]`;
      }
      setPreviewValues(values);
    }
  }, [selectedName]);

  // 提取 {{placeholder}} 占位符
  function extractPlaceholders(text: string): string[] {
    const regex = /\{\{([^}]+)\}\}/g;
    const matches: string[] = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
      if (!matches.includes(match[1])) {
        matches.push(match[1]);
      }
    }
    return matches;
  }

  // 替换占位符生成预览
  function getPreviewContent(): string {
    let content = editContent;
    for (const [key, value] of Object.entries(previewValues)) {
      content = content.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), value);
    }
    return content;
  }

  // 保存
  const handleSave = useCallback(async () => {
    if (!selectedName) return;
    setSaving(true);
    try {
      await api.updatePrompt(selectedName, editContent);
      toast({ title: "保存成功", description: `提示词模板 "${selectedName}" 已更新` });
      loadTemplates();
    } catch (err) {
      toast({
        title: "保存失败",
        description: err instanceof Error ? err.message : "未知错误",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  }, [selectedName, editContent, toast, loadTemplates]);

  // 重置
  const handleReset = useCallback(async () => {
    if (!selectedName) return;
    try {
      await api.resetPrompt(selectedName);
      toast({ title: "重置成功", description: `提示词模板 "${selectedName}" 已恢复默认` });
      loadTemplates();
    } catch (err) {
      toast({
        title: "重置失败",
        description: err instanceof Error ? err.message : "未知错误",
        variant: "destructive",
      });
    }
  }, [selectedName, toast, loadTemplates]);

  // 高亮占位符的渲染
  function renderHighlightedContent(content: string) {
    const parts = content.split(/(\{\{[^}]+\}\})/g);
    return parts.map((part, i) => {
      if (part.match(/^\{\{[^}]+\}\}$/)) {
        return (
          <span
            key={i}
            className="bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200 px-1 rounded text-xs font-mono"
          >
            {part}
          </span>
        );
      }
      return <span key={i}>{part}</span>;
    });
  }

  const isModified =
    selectedTemplate && editContent !== selectedTemplate.template;

  return (
    <div className="flex h-[calc(100vh-44px)]">
      {/* 左侧模板列表 */}
      <div className="w-72 border-r bg-muted/30 shrink-0 flex flex-col">
        <div className="p-3">
          <h2 className="text-sm font-semibold mb-1">提示词模板</h2>
          <p className="text-[10px] text-muted-foreground">
            共 {templates.length} 个模板
          </p>
        </div>
        <ScrollArea className="flex-1">
          <div className="px-3 pb-3 space-y-0.5">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : templates.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-8">
                暂无模板
              </p>
            ) : (
              templates.map((t) => (
                <button
                  key={t.name}
                  className={`w-full text-left text-xs px-2.5 py-2 rounded-md hover:bg-muted transition-colors ${
                    selectedName === t.name
                      ? "bg-secondary text-secondary-foreground"
                      : ""
                  }`}
                  onClick={() => setSelectedName(t.name)}
                >
                  <div className="flex items-center gap-1.5">
                    <Code2 className="h-3 w-3 shrink-0" />
                    <span className="font-medium truncate">{t.name}</span>
                    {t.is_overridden && (
                      <Badge
                        variant="secondary"
                        className="text-[9px] px-1 py-0 ml-auto shrink-0"
                      >
                        已覆盖
                      </Badge>
                    )}
                  </div>
                  <p className="text-muted-foreground mt-0.5 truncate text-[10px]">
                    {templateDescriptions[t.name] || t.name}
                  </p>
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* 右侧编辑区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {selectedTemplate ? (
          <>
            {/* 编辑器头部 */}
            <div className="flex items-center justify-between px-4 py-2 border-b">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold">{selectedTemplate.name}</h3>
                {selectedTemplate.is_overridden && (
                  <Badge variant="outline" className="text-[10px]">
                    已自定义
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="h-7 text-xs gap-1">
                      <Eye className="h-3 w-3" />
                      预览
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl max-h-[80vh]">
                    <DialogHeader>
                      <DialogTitle>模板预览 - {selectedTemplate.name}</DialogTitle>
                    </DialogHeader>
                    <ScrollArea className="max-h-[60vh]">
                      <div className="space-y-4">
                        {/* 占位符替换 */}
                        {extractPlaceholders(editContent).length > 0 && (
                          <div className="space-y-2">
                            <p className="text-xs text-muted-foreground font-medium">
                              替换占位符:
                            </p>
                            {extractPlaceholders(editContent).map((p) => (
                              <div key={p} className="flex items-center gap-2">
                                <code className="text-xs bg-muted px-1.5 py-0.5 rounded">
                                  {`{{${p}}}`}
                                </code>
                                <Input
                                  value={previewValues[p] || ""}
                                  onChange={(e) =>
                                    setPreviewValues((prev) => ({
                                      ...prev,
                                      [p]: e.target.value,
                                    }))
                                  }
                                  className="h-7 text-xs"
                                  placeholder={`输入 ${p} 的值`}
                                />
                              </div>
                            ))}
                            <Separator className="my-3" />
                          </div>
                        )}
                        {/* 预览结果 */}
                        <div>
                          <p className="text-xs text-muted-foreground font-medium mb-2">
                            预览结果:
                          </p>
                          <div className="bg-muted/50 rounded-lg p-4 text-sm whitespace-pre-wrap">
                            {getPreviewContent()}
                          </div>
                        </div>
                      </div>
                    </ScrollArea>
                  </DialogContent>
                </Dialog>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs gap-1"
                  onClick={handleReset}
                >
                  <RotateCcw className="h-3 w-3" />
                  重置
                </Button>
                <Button
                  size="sm"
                  className="h-7 text-xs gap-1"
                  onClick={handleSave}
                  disabled={saving || !isModified}
                >
                  {saving ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <Save className="h-3 w-3" />
                  )}
                  保存
                </Button>
              </div>
            </div>

            {/* 编辑器 */}
            <div className="flex-1 flex">
              <div className="flex-1 p-4">
                <Textarea
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  className="h-full font-mono text-xs resize-none leading-relaxed"
                  placeholder="模板内容..."
                />
              </div>
              <div className="w-px bg-border" />
              <div className="flex-1 p-4 overflow-auto">
                <div className="text-xs whitespace-pre-wrap leading-relaxed">
                  {renderHighlightedContent(editContent)}
                </div>
              </div>
            </div>

            {/* 底部信息 */}
            <div className="px-4 py-1.5 border-t text-[10px] text-muted-foreground flex items-center gap-3">
              <span>占位符: {extractPlaceholders(editContent).length} 个</span>
              <span>字符: {editContent.length}</span>
              <span>行数: {editContent.split("\n").length}</span>
              {isModified && (
                <span className="text-amber-500 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  未保存更改
                </span>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Code2 className="h-12 w-12 text-muted-foreground/30 mx-auto mb-4" />
              <h2 className="text-lg font-semibold mb-2">提示词编辑器</h2>
              <p className="text-sm text-muted-foreground">
                选择左侧模板开始编辑
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
