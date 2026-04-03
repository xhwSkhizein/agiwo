"use client";

import { useId } from "react";
import { Send } from "lucide-react";

type ChatInputBarProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder?: string;
  label?: string;
  submitLabel?: string;
};

export function ChatInputBar({
  value,
  onChange,
  onSubmit,
  disabled = false,
  placeholder = "Type a message...",
  label = "Message",
  submitLabel = "Send message",
}: ChatInputBarProps) {
  const inputId = useId();

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex items-center gap-3"
    >
      <div className="min-w-0 flex-1">
        <label htmlFor={inputId} className="sr-only">
          {label}
        </label>
        <input
          id={inputId}
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          disabled={disabled}
          className="ui-input"
        />
      </div>
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        aria-label={submitLabel}
        className="ui-button ui-button-primary ui-button-icon"
      >
        <Send className="w-4 h-4" />
      </button>
    </form>
  );
}

export default ChatInputBar;
