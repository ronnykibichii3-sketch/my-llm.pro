import { apiClient } from '../lib/api-client';
import type { MessageCreate, MessageRead } from '../types/chat';

export const chatService = {
  async send(conversationId: number, payload: MessageCreate): Promise<MessageRead> {
    const { data } = await apiClient.post<MessageRead>(`/conversations/${conversationId}/chat`, payload);
    return data;
  },
};
