/** Single page-level store for messages, domain, and the active source. */

import { create } from "zustand";
import type { ChatMessage, DomainFilter } from "./types";

export interface ActiveSource {
  messageId: string;
  n: number;
}

interface ChatState {
  messages: ChatMessage[];
  domain: DomainFilter;
  activeSource: ActiveSource | null;
  hydrated: boolean;

  setDomain: (domain: DomainFilter) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (
    id: string,
    patch: Partial<Omit<ChatMessage, "id">>,
  ) => void;
  appendToken: (id: string, text: string) => void;
  removeMessage: (id: string) => void;
  selectSource: (messageId: string, n: number) => void;
  clearSource: () => void;
  reset: () => void;
  hydrate: (messages: ChatMessage[], domain: DomainFilter) => void;
}

export const useChatStore = create<ChatState>()((set) => ({
  messages: [],
  domain: "all",
  activeSource: null,
  hydrated: false,

  setDomain: (domain) => set({ domain }),

  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  updateMessage: (id, patch) =>
    set((state) => ({
      messages: state.messages.map((message) =>
        message.id === id ? { ...message, ...patch } : message,
      ),
    })),

  appendToken: (id, text) =>
    set((state) => ({
      messages: state.messages.map((message) =>
        message.id === id
          ? { ...message, content: message.content + text }
          : message,
      ),
    })),

  removeMessage: (id) =>
    set((state) => ({
      messages: state.messages.filter((message) => message.id !== id),
      activeSource:
        state.activeSource?.messageId === id ? null : state.activeSource,
    })),

  selectSource: (messageId, n) => set({ activeSource: { messageId, n } }),

  clearSource: () => set({ activeSource: null }),

  reset: () =>
    set({ messages: [], activeSource: null, hydrated: true }),

  hydrate: (messages, domain) =>
    set({ messages, domain, activeSource: null, hydrated: true }),
}));
