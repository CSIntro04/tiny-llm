import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

// shadcn 约定的工具函数：合并 className，后面的 Tailwind 类覆盖前面冲突的类
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
