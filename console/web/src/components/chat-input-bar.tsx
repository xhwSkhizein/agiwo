"use client";

import { Send } from "lucide-react";

type ChatInputBarProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder?: string;
};

export function ChatInputBar({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = "Type a message...",
}: ChatInputBarProps) {
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex items-center gap-3"
    >
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="flex-1 px-4 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-sm focus:outline-none focus:border-zinc-500 disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="p-2.5 rounded-lg bg-white text-black hover:bg-zinc-200 transition-colors disabled:opacity-30"
      >
        <Send className="w-4 h-4" />
      </button>
    </form>
  );
}

export default ChatInputBar;
