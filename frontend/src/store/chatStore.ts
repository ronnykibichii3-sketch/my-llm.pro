import { create } from 'zustand';

import { messageService } from '../services/message.service';
import type { MessageRead } from '../types/chat';

interface ChatState {
  conversationId: number | null;
  messages: MessageRead[];
  isLoading: boolean;
  error: string | null;
  setConversationId: (conversationId: number | null) => void;
  loadMessages: (conversationId: number) => Promise<void>;
  appendMessage: (message: MessageRead) => void;
  removeMessage: (messageId: number) => void;
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

const toErrorMessage = (caught: unknown): string =>
  caught instanceof Error ? caught.message : 'Failed to load messages';

export const useChatStore = create<ChatState>((set, get) => ({
  conversationId: null,
  messages: [],
  isLoading: false,
  error: null,
  setConversationId: (conversationId) => set({ conversationId, messages: [], error: null }),
  loadMessages: async (conversationId) => {
    set({ isLoading: true, error: null });
    try {
      const messages = await messageService.list(conversationId);
      if (get().conversationId === conversationId) {
        set({ messages, isLoading: false });
      }
    } catch (caught) {
      if (get().conversationId === conversationId) {
        set({ isLoading: false, error: toErrorMessage(caught) });
      }
    }
  },
  appendMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  removeMessage: (messageId) =>
    set((state) => ({ messages: state.messages.filter((message) => message.id !== messageId) })),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  reset: () => set({ conversationId: null, messages: [], isLoading: false, error: null }),
}));
