"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, BookOpen, Code2, Settings, Moon, Sun, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "next-themes";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const navItems = [
  { href: "/", label: "智能问答", icon: MessageSquare },
  { href: "/wiki", label: "Wiki 浏览", icon: BookOpen },
  { href: "/prompts", label: "提示词编辑", icon: Code2 },
  { href: "/settings", label: "系统设置", icon: Settings },
];

export function AppNav() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();

  return (
    <TooltipProvider delayDuration={0}>
      <nav className="flex items-center gap-1 px-2">
        <div className="flex items-center gap-1.5 mr-2">
          <Zap className="h-5 w-5 text-primary" />
          <span className="font-semibold text-sm hidden sm:inline">Local RAG</span>
        </div>
        <div className="h-5 w-px bg-border mx-1" />
        {navItems.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Tooltip key={item.href}>
              <TooltipTrigger asChild>
                <Link href={item.href}>
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    size="sm"
                    className={`gap-1.5 text-xs ${
                      isActive
                        ? "bg-secondary text-secondary-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <item.icon className="h-3.5 w-3.5" />
                    <span className="hidden md:inline">{item.label}</span>
                  </Button>
                </Link>
              </TooltipTrigger>
              <TooltipContent>{item.label}</TooltipContent>
            </Tooltip>
          );
        })}
        <div className="h-5 w-px bg-border mx-1" />
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? (
                <Sun className="h-3.5 w-3.5" />
              ) : (
                <Moon className="h-3.5 w-3.5" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            {theme === "dark" ? "切换亮色模式" : "切换暗色模式"}
          </TooltipContent>
        </Tooltip>
      </nav>
    </TooltipProvider>
  );
}
