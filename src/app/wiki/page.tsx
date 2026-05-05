"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  BookOpen,
  Search,
  Sparkles,
  Download,
  Network,
  ChevronDown,
  Loader2,
  FileText,
  Layers,
  Lightbulb,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Label } from "@/components/ui/label";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { MermaidRenderer } from "@/components/mermaid-renderer";
import { api } from "@/lib/api";
import type { WikiPageSummary, WikiPage, WikiPageType } from "@/lib/types";
import { useToast } from "@/hooks/use-toast";

const pageTypeLabels: Record<string, string> = {
  index: "索引",
  summary: "摘要",
  entity: "实体",
  concept: "概念",
  synthesis: "综合",
};

const pageTypeIcons: Record<string, React.ElementType> = {
  index: Layers,
  summary: FileText,
  entity: BookOpen,
  concept: Lightbulb,
  synthesis: Sparkles,
};

export default function WikiBrowserPage() {
  const { toast } = useToast();
  const [pages, setPages] = useState<WikiPageSummary[]>([]);
  const [selectedPage, setSelectedPage] = useState<WikiPage | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [pageLoading, setPageLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generateDialogOpen, setGenerateDialogOpen] = useState(false);
  const [granularity, setGranularity] = useState<string>("standard");
  const [mermaidChart, setMermaidChart] = useState<string>("");
  const [showGraph, setShowGraph] = useState(false);
  const [buildingGraph, setBuildingGraph] = useState(false);

  // 分组
  const groupedPages = React.useMemo(() => {
    const filtered = pages.filter((p) => {
      const matchesType = filterType === "all" || p.type === filterType;
      const matchesSearch =
        !searchQuery ||
        p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.slug.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesType && matchesSearch;
    });

    const groups: Record<string, WikiPageSummary[]> = {};
    for (const p of filtered) {
      const type = p.type || "other";
      if (!groups[type]) groups[type] = [];
      groups[type].push(p);
    }
    return groups;
  }, [pages, filterType, searchQuery]);

  // 加载 Wiki 页面列表
  const loadPages = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.listWikiPages();
      setPages(res.pages);
    } catch (err) {
      toast({
        title: "加载失败",
        description: err instanceof Error ? err.message : "无法加载Wiki页面",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadPages();
  }, [loadPages]);

  // 加载页面详情
  const loadPageDetail = useCallback(
    async (slug: string) => {
      setPageLoading(true);
      try {
        const page = await api.getWikiPage(slug);
        setSelectedPage(page);
      } catch (err) {
        toast({
          title: "加载失败",
          description: err instanceof Error ? err.message : "无法加载页面",
          variant: "destructive",
        });
      } finally {
        setPageLoading(false);
      }
    },
    [toast]
  );

  // 生成 Wiki
  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    try {
      await api.generateWiki({
        granularity: granularity as "focused" | "standard" | "exhaustive",
      });
      toast({ title: "Wiki 生成成功" });
      setGenerateDialogOpen(false);
      loadPages();
    } catch (err) {
      toast({
        title: "生成失败",
        description: err instanceof Error ? err.message : "未知错误",
        variant: "destructive",
      });
    } finally {
      setGenerating(false);
    }
  }, [granularity, toast, loadPages]);

  // 构建知识图谱
  const handleBuildGraph = useCallback(async () => {
    setBuildingGraph(true);
    try {
      const res = await api.buildKnowledgeGraph();
      setMermaidChart(res.mermaid);
      setShowGraph(true);
      toast({
        title: "知识图谱构建完成",
        description: `${res.entities} 个实体, ${res.relationships} 个关系`,
      });
    } catch (err) {
      toast({
        title: "构建失败",
        description: err instanceof Error ? err.message : "未知错误",
        variant: "destructive",
      });
    } finally {
      setBuildingGraph(false);
    }
  }, [toast]);

  // 导出 Wiki
  const handleExport = useCallback(async () => {
    try {
      const res = await api.exportWiki();
      toast({ title: "导出成功", description: res.message });
    } catch (err) {
      toast({
        title: "导出失败",
        description: err instanceof Error ? err.message : "未知错误",
        variant: "destructive",
      });
    }
  }, [toast]);

  // 加载图谱
  const loadMermaid = useCallback(async () => {
    try {
      const res = await api.getGraphMermaid();
      if (res.mermaid) {
        setMermaidChart(res.mermaid);
      }
    } catch {
      // 静默
    }
  }, []);

  useEffect(() => {
    loadMermaid();
  }, [loadMermaid]);

  // Wiki 链接点击
  const handleWikiLinkClick = useCallback(
    (slug: string) => {
      loadPageDetail(slug);
    },
    [loadPageDetail]
  );

  return (
    <div className="flex h-[calc(100vh-44px)]">
      {/* 左侧 Wiki 页面列表 */}
      <div className="w-72 border-r bg-muted/30 shrink-0 flex flex-col">
        <div className="p-3 space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Wiki 页面</h2>
            <div className="flex items-center gap-1">
              <Dialog open={generateDialogOpen} onOpenChange={setGenerateDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
                    <Sparkles className="h-3 w-3 mr-1" />
                    生成
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>生成 Wiki 知识库</DialogTitle>
                    <DialogDescription>
                      从已上传文档生成 Wiki 页面，请选择生成粒度
                    </DialogDescription>
                  </DialogHeader>
                  <div className="space-y-3 py-2">
                    <Label className="text-sm font-medium">生成粒度</Label>
                    <Select value={granularity} onValueChange={setGranularity}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="focused">
                          聚焦 - 仅提取核心实体和概念
                        </SelectItem>
                        <SelectItem value="standard">
                          标准 - 提取主要实体和概念
                        </SelectItem>
                        <SelectItem value="exhaustive">
                          详尽 - 提取所有可能的实体和概念
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      onClick={() => setGenerateDialogOpen(false)}
                    >
                      取消
                    </Button>
                    <Button onClick={handleGenerate} disabled={generating}>
                      {generating && <Loader2 className="h-4 w-4 mr-1 animate-spin" />}
                      生成
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="搜索页面..."
              className="pl-8 h-8 text-xs"
            />
          </div>

          <Select value={filterType} onValueChange={setFilterType}>
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="筛选类型" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">全部类型</SelectItem>
              <SelectItem value="summary">摘要</SelectItem>
              <SelectItem value="entity">实体</SelectItem>
              <SelectItem value="concept">概念</SelectItem>
              <SelectItem value="index">索引</SelectItem>
              <SelectItem value="synthesis">综合</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <ScrollArea className="flex-1">
          <div className="px-3 pb-3 space-y-2">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            ) : Object.keys(groupedPages).length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-8">
                {pages.length === 0 ? "暂无 Wiki 页面，请先生成" : "没有匹配的页面"}
              </p>
            ) : (
              Object.entries(groupedPages).map(([type, typePages]) => {
                const Icon = pageTypeIcons[type] || FileText;
                return (
                  <Collapsible key={type} defaultOpen>
                    <CollapsibleTrigger className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground w-full py-1 hover:text-foreground">
                      <ChevronDown className="h-3 w-3" />
                      <Icon className="h-3 w-3" />
                      <span>{pageTypeLabels[type] || type}</span>
                      <Badge variant="secondary" className="text-[10px] ml-auto px-1 py-0">
                        {typePages.length}
                      </Badge>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="pl-2 space-y-0.5 mt-0.5">
                      {typePages.map((page) => (
                        <button
                          key={page.slug}
                          className={`w-full text-left text-xs px-2 py-1.5 rounded-md hover:bg-muted transition-colors truncate ${
                            selectedPage?.slug === page.slug
                              ? "bg-secondary text-secondary-foreground"
                              : ""
                          }`}
                          onClick={() => loadPageDetail(page.slug)}
                        >
                          {page.title}
                        </button>
                      ))}
                    </CollapsibleContent>
                  </Collapsible>
                );
              })
            )}
          </div>
        </ScrollArea>

        <div className="p-3 border-t space-y-1">
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-xs"
            onClick={handleBuildGraph}
            disabled={buildingGraph}
          >
            {buildingGraph ? (
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
            ) : (
              <Network className="h-3.5 w-3.5 mr-1.5" />
            )}
            构建知识图谱
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-xs"
            onClick={handleExport}
          >
            <Download className="h-3.5 w-3.5 mr-1.5" />
            导出 Wiki
          </Button>
        </div>
      </div>

      {/* 右侧内容区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {showGraph && mermaidChart && (
          <div className="border-b">
            <div className="flex items-center justify-between px-4 py-2 bg-muted/30">
              <span className="text-xs font-medium flex items-center gap-1.5">
                <Network className="h-3.5 w-3.5" />
                知识图谱可视化
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs"
                onClick={() => setShowGraph(false)}
              >
                关闭
              </Button>
            </div>
            <MermaidRenderer chart={mermaidChart} className="max-h-64 p-2" />
          </div>
        )}

        <ScrollArea className="flex-1">
          <div className="max-w-4xl mx-auto p-6">
            {pageLoading ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : selectedPage ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {pageTypeLabels[selectedPage.page_type] || selectedPage.page_type}
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    {selectedPage.status}
                  </Badge>
                  <h1 className="text-xl font-bold">{selectedPage.title}</h1>
                </div>
                <MarkdownRenderer
                  content={selectedPage.content}
                  onWikiLinkClick={handleWikiLinkClick}
                />
                {selectedPage.out_links.length > 0 && (
                  <div className="pt-4 border-t">
                    <p className="text-xs text-muted-foreground mb-2">相关页面</p>
                    <div className="flex flex-wrap gap-1">
                      {selectedPage.out_links.map((link) => (
                        <Badge
                          key={link}
                          variant="outline"
                          className="text-xs cursor-pointer hover:bg-muted"
                          onClick={() => loadPageDetail(link)}
                        >
                          {link}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <BookOpen className="h-12 w-12 text-muted-foreground/30 mb-4" />
                <h2 className="text-lg font-semibold mb-2">Wiki 知识库</h2>
                <p className="text-sm text-muted-foreground max-w-md">
                  选择左侧页面查看内容，或点击"生成"按钮从文档创建 Wiki 知识库。
                </p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
