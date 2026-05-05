"use client";

import React from "react";
import { User, Bot, ChevronDown, ChevronRight, Wrench, Eye, Loader2 } from "lucide-react";
import { MarkdownRenderer } from "./markdown-renderer";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { ChatMessage as ChatMessageType, AgentStep } from "@/lib/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

function AgentStepView({ step }: { step: AgentStep }) {
  const [open, setOpen] = React.useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground w-full py-1">
        {open ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <span className="font-medium">步骤 {step.step_index + 1}</span>
        {step.thought && (
          <span className="truncate text-muted-foreground/70">
            - {step.thought.slice(0, 60)}
            {step.thought.length > 60 ? "..." : ""}
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent className="pl-4 space-y-2 border-l-2 border-muted ml-1 mt-1">
        {step.thought && (
          <div className="text-xs">
            <span className="text-muted-foreground font-medium">思考: </span>
            <span className="text-foreground/80">{step.thought}</span>
          </div>
        )}
        {step.tool_calls.length > 0 && (
          <div className="space-y-1">
            {step.tool_calls.map((tc) => (
              <div
                key={tc.call_id}
                className="flex items-start gap-1.5 text-xs bg-muted/50 rounded p-1.5"
              >
                <Wrench className="h-3 w-3 shrink-0 mt-0.5 text-amber-500" />
                <div>
                  <span className="font-medium">{tc.name}</span>
                  <pre className="text-[10px] text-muted-foreground mt-0.5 whitespace-pre-wrap">
                    {JSON.stringify(tc.arguments, null, 2)}
                  </pre>
                </div>
              </div>
            ))}
          </div>
        )}
        {step.tool_results.length > 0 && (
          <div className="space-y-1">
            {step.tool_results.map((tr, i) => (
              <div
                key={tr.call_id || i}
                className="flex items-start gap-1.5 text-xs bg-muted/30 rounded p-1.5"
              >
                <Eye className="h-3 w-3 shrink-0 mt-0.5 text-emerald-500" />
                <div>
                  <span className="font-medium">{tr.name}</span>
                  {tr.is_error ? (
                    <span className="text-destructive ml-1">错误: {tr.error}</span>
                  ) : (
                    <pre className="text-[10px] text-muted-foreground mt-0.5 whitespace-pre-wrap max-h-24 overflow-auto">
                      {tr.output.slice(0, 500)}
                      {tr.output.length > 500 ? "..." : ""}
                    </pre>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        {step.observation && (
          <div className="text-xs">
            <span className="text-muted-foreground font-medium">观察: </span>
            <span className="text-foreground/80">{step.observation}</span>
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="shrink-0 h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center mt-0.5">
          <Bot className="h-4 w-4 text-primary" />
        </div>
      )}
      <div className={`max-w-[80%] space-y-2 ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted"
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <>
              {message.isStreaming && !message.content ? (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span className="text-xs">思考中...</span>
                </div>
              ) : (
                <MarkdownRenderer content={message.content} />
              )}
            </>
          )}
        </div>

        {/* Agent 步骤 */}
        {!isUser && message.agentSteps && message.agentSteps.length > 0 && (
          <div className="space-y-0.5">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Agent 推理过程
            </div>
            {message.agentSteps.map((step, i) => (
              <AgentStepView key={i} step={step} />
            ))}
          </div>
        )}

        {/* 来源引用 */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="space-y-1">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
              来源引用
            </div>
            <div className="flex flex-wrap gap-1">
              {message.sources.map((src, i) => (
                <Badge
                  key={i}
                  variant="outline"
                  className="text-[10px] py-0 px-1.5"
                >
                  {src.chunk.metadata?.section_title ||
                    src.chunk.metadata?.filename ||
                    `来源 ${i + 1}`}
                  <span className="ml-1 text-muted-foreground">
                    {src.score.toFixed(2)}
                  </span>
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* 推荐问题 */}
        {!isUser && message.recommendedQuestions && message.recommendedQuestions.length > 0 && (
          <div className="space-y-1">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider">
              推荐问题
            </div>
            <div className="flex flex-wrap gap-1">
              {message.recommendedQuestions.map((q, i) => (
                <Badge
                  key={i}
                  variant="secondary"
                  className="text-xs py-1 px-2 cursor-pointer hover:bg-secondary/80 transition-colors"
                  onClick={() => {
                    // 通过自定义事件触发问题点击
                    const event = new CustomEvent("recommended-question-click", {
                      detail: q,
                    });
                    window.dispatchEvent(event);
                  }}
                >
                  {q}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>
      {isUser && (
        <div className="shrink-0 h-7 w-7 rounded-full bg-muted flex items-center justify-center mt-0.5">
          <User className="h-4 w-4 text-muted-foreground" />
        </div>
      )}
    </div>
  );
}
