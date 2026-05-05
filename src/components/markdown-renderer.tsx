"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

interface MarkdownRendererProps {
  content: string;
  className?: string;
  onWikiLinkClick?: (slug: string) => void;
}

export function MarkdownRenderer({
  content,
  className = "",
  onWikiLinkClick,
}: MarkdownRendererProps) {
  // 处理 [[slug|标题]] 格式的 Wiki 链接
  const processContent = (text: string): string => {
    return text.replace(
      /\[\[([^|]+)\|([^\]]+)\]\]/g,
      '<a href="#" data-wiki-link="$1" class="text-primary underline underline-offset-2 hover:text-primary/80 cursor-pointer">$2</a>'
    );
  };

  const processedContent = processContent(content);

  return (
    <div
      className={`prose prose-sm dark:prose-invert max-w-none ${className}`}
      onClick={(e) => {
        const target = e.target as HTMLElement;
        const wikiLink = target.closest("[data-wiki-link]");
        if (wikiLink && onWikiLinkClick) {
          e.preventDefault();
          const slug = wikiLink.getAttribute("data-wiki-link");
          if (slug) onWikiLinkClick(slug);
        }
      }}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}
