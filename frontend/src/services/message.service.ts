import { apiClient } from '../lib/api-client';
import type { MessageCreate, MessageRead } from '../types/chat';

export const messageService = {
  async list(conversationId: number): Promise<MessageRead[]> {
    const { data } = await apiClient.get<MessageRead[]>(`/conversations/${conversationId}/messages`);
    return data;
  },

  async create(conversationId: number, payload: MessageCreate): Promise<MessageRead> {
    const { data } = await apiClient.post<MessageRead>(`/conversations/${conversationId}/messages`, payload);
    return data;
  },
};