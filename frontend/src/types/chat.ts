export type ChatMessageRole = 'user' | 'assistant';

export interface MessageCreate {
  content: string;
}

export interface MessageRead {
  id: number;
  role: ChatMessageRole;
  content: string;
  created_at: string;
}
