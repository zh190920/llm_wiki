"use client";

import React, { useCallback, useState } from "react";
import {
  Upload,
  FileText,
  Trash2,
  MessageSquare,
  Bot,
  BookOpen,
  Settings2,
  ChevronDown,
  ChevronRight,
  Loader2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ChatMode, DocumentMetadata } from "@/lib/types";
import { api } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

interface ChatSidebarProps {
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  documents: DocumentMetadata[];
  onDocumentsChange: () => void;
  similarityThreshold: number;
  onSimilarityThresholdChange: (value: number) => void;
  topK: number;
  onTopKChange: (value: number) => void;
  deepMode: boolean;
  onDeepModeChange: (value: boolean) => void;
  hierarchicalChunking: boolean;
  onHierarchicalChunkingChange: (value: boolean) => void;
  onClearHistory: () => void;
}

export function ChatSidebar({
  mode,
  onModeChange,
  documents,
  onDocumentsChange,
  similarityThreshold,
  onSimilarityThresholdChange,
  topK,
  onTopKChange,
  deepMode,
  onDeepModeChange,
  hierarchicalChunking,
  onHierarchicalChunkingChange,
  onClearHistory,
}: ChatSidebarProps) {
  const { toast } = useToast();
  const [uploading, setUploading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = useCallback(
    async (files: FileList | File[]) => {
      const fileArray = Array.from(files);
      if (fileArray.length === 0) return;

      setUploading(true);
      try {
        for (const file of fileArray) {
          await api.uploadDocument(file);
        }
        toast({ title: "上传成功", description: `已上传 ${fileArray.length} 个文件` });
        onDocumentsChange();
      } catch (err) {
        toast({
          title: "上传失败",
          description: err instanceof Error ? err.message : "未知错误",
          variant: "destructive",
        });
      } finally {
        setUploading(false);
      }
    },
    [toast, onDocumentsChange]
  );

  const handleDelete = useCallback(
    async (docId: string) => {
      try {
        await api.deleteDocument(docId);
        toast({ title: "删除成功" });
        onDocumentsChange();
      } catch (err) {
        toast({
          title: "删除失败",
          description: err instanceof Error ? err.message : "未知错误",
          variant: "destructive",
        });
      }
    },
    [toast, onDocumentsChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files);
      }
    },
    [handleUpload]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        handleUpload(e.target.files);
        e.target.value = "";
      }
    },
    [handleUpload]
  );

  const modes: { value: ChatMode; label: string; icon: React.ElementType; desc: string }[] = [
    { value: "rag", label: "RAG快速问答", icon: MessageSquare, desc: "基于检索的快速回答" },
    { value: "agent", label: "Agent智能推理", icon: Bot, desc: "多步推理与工具调用" },
    { value: "wiki", label: "Wiki模式", icon: BookOpen, desc: "查询Wiki知识库" },
  ];

  return (
    <div className="flex flex-col h-full w-full">
      {/* 文档上传区域 */}
      <div className="p-3 space-y-2">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          文档管理
        </h3>
        <div
          className={`border-2 border-dashed rounded-lg p-3 text-center transition-colors cursor-pointer ${
            dragOver
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-muted-foreground/50"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("file-upload")?.click()}
        >
          {uploading ? (
            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              上传中...
            </div>
          ) : (
            <>
              <Upload className="h-5 w-5 mx-auto mb-1 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">
                拖拽文件到此处或点击上传
              </p>
              <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                支持 PDF、Markdown
              </p>
            </>
          )}
          <input
            id="file-upload"
            type="file"
            className="hidden"
            accept=".pdf,.md,.markdown,.txt"
            multiple
            onChange={handleFileInput}
          />
        </div>
      </div>

      {/* 文档列表 */}
      <div className="px-3 pb-2">
        <ScrollArea className="max-h-36">
          {documents.length === 0 ? (
            <p className="text-xs text-muted-foreground/60 text-center py-2">
              暂无文档
            </p>
          ) : (
            <div className="space-y-1">
              {documents.map((doc) => (
                <div
                  key={doc.doc_id}
                  className="flex items-center gap-1.5 text-xs p-1.5 rounded-md hover:bg-muted group"
                >
                  <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate flex-1">{doc.filename}</span>
                  <Badge variant="secondary" className="text-[10px] px-1 py-0 shrink-0">
                    {doc.chunk_count}块
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 shrink-0"
                    onClick={() => handleDelete(doc.doc_id)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>

      <Separator />

      {/* 聊天模式选择 */}
      <div className="p-3 space-y-2">
        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          聊天模式
        </h3>
        <div className="space-y-1">
          {modes.map((m) => (
            <Button
              key={m.value}
              variant={mode === m.value ? "secondary" : "ghost"}
              size="sm"
              className={`w-full justify-start gap-2 text-xs h-8 ${
                mode === m.value ? "bg-secondary" : ""
              }`}
              onClick={() => onModeChange(m.value)}
            >
              <m.icon className="h-3.5 w-3.5" />
              <span>{m.label}</span>
            </Button>
          ))}
        </div>
      </div>

      <Separator />

      {/* 检索设置 */}
      <div className="p-3 space-y-2">
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-between text-xs h-7 px-0"
          onClick={() => setShowSettings(!showSettings)}
        >
          <span className="font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
            <Settings2 className="h-3 w-3" />
            检索设置
          </span>
          {showSettings ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </Button>

        {showSettings && (
          <div className="space-y-3 pt-1">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-xs">相似度阈值</Label>
                <span className="text-[10px] text-muted-foreground">
                  {similarityThreshold.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[similarityThreshold]}
                min={0}
                max={1}
                step={0.05}
                onValueChange={([v]) => onSimilarityThresholdChange(v)}
                className="py-1"
              />
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">Top-K</Label>
              <Select
                value={String(topK)}
                onValueChange={(v) => onTopKChange(Number(v))}
              >
                <SelectTrigger className="h-7 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {[1, 3, 5, 10, 15, 20].map((k) => (
                    <SelectItem key={k} value={String(k)}>
                      {k}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">深度模式</Label>
              <Switch checked={deepMode} onCheckedChange={onDeepModeChange} />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-xs">层级分块</Label>
              <Switch
                checked={hierarchicalChunking}
                onCheckedChange={onHierarchicalChunkingChange}
              />
            </div>
          </div>
        )}
      </div>

      <div className="mt-auto p-3">
        <Separator className="mb-3" />
        <Button
          variant="ghost"
          size="sm"
          className="w-full text-xs text-muted-foreground hover:text-destructive"
          onClick={onClearHistory}
        >
          <Trash2 className="h-3.5 w-3.5 mr-1.5" />
          清空对话历史
        </Button>
      </div>
    </div>
  );
}
