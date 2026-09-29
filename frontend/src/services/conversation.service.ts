import { apiClient } from '../lib/api-client';
import type { ConversationCreate, ConversationRead } from '../types/conversation';

export const conversationService = {
  async list(): Promise<ConversationRead[]> {
    const { data } = await apiClient.get<ConversationRead[]>('/conversations');
    return data;
  },

  async create(payload: ConversationCreate): Promise<ConversationRead> {
    const { data } = await apiClient.post<ConversationRead>('/conversations', payload);
    return data;
  },
};
