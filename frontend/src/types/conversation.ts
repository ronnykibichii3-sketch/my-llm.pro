export interface ConversationCreate {
  title: string;
}

export interface ConversationRead {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
}
