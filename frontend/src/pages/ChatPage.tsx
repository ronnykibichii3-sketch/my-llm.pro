import { useEffect, useRef, useState, type FormEvent } from 'react';

import { chatService } from '../services/chat.service';
import { useChatStore } from '../store/chatStore';

const MAX_MESSAGE_LENGTH = 20_000;

let optimisticIdCounter = 0;

const ChatPage = () => {
  const [draft, setDraft] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const conversationId = useChatStore((state) => state.conversationId);
  const messages = useChatStore((state) => state.messages);
  const isLoading = useChatStore((state) => state.isLoading);
  const error = useChatStore((state) => state.error);
  const loadMessages = useChatStore((state) => state.loadMessages);
  const appendMessage = useChatStore((state) => state.appendMessage);
  const removeMessage = useChatStore((state) => state.removeMessage);
  const setLoading = useChatStore((state) => state.setLoading);
  const setError = useChatStore((state) => state.setError);

  useEffect(() => {
    if (conversationId !== null) {
      void loadMessages(conversationId);
    }
  }, [conversationId, loadMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const content = draft.trim();
    if (content.length === 0 || isLoading || conversationId === null) {
      return;
    }

    const optimisticId = -(Date.now() * 10 + optimisticIdCounter++);
    appendMessage({
      id: optimisticId,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    });
    setDraft('');
    setLoading(true);
    setError(null);

    try {
      await chatService.send(conversationId, { content });
    } catch (caught) {
      removeMessage(optimisticId);
      setError(caught instanceof Error ? caught.message : 'Failed to send message');
      setLoading(false);
      return;
    }

    await loadMessages(conversationId);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {conversationId === null && (
          <p className="text-center text-sm text-slate-500">
            Select or create a conversation to start chatting.
          </p>
        )}

        {conversationId !== null && messages.length === 0 && !isLoading && !error && (
          <p className="text-center text-sm text-slate-500">Send a message to start the conversation.</p>
        )}

        {conversationId !== null && isLoading && messages.length === 0 && (
          <p className="text-center text-sm text-slate-500">Loading messages…</p>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[75%] whitespace-pre-wrap rounded-lg px-4 py-2 text-sm ${
                message.role === 'user' ? 'bg-primary/20 text-slate-100' : 'bg-surface text-slate-300'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}

        {isLoading && messages.length > 0 && (
          <div className="flex justify-start">
            <div className="rounded-lg bg-surface px-4 py-2 text-sm italic text-slate-500">Thinking…</div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="border-t border-slate-800 p-4">
        {error && (
          <div className="mb-2 rounded border border-rose-800 bg-rose-950 px-3 py-2 text-xs text-rose-400" role="alert">
            {error}
          </div>
        )}

        <form onSubmit={handleSend} className="flex items-end gap-2">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            rows={2}
            maxLength={MAX_MESSAGE_LENGTH}
            disabled={conversationId === null}
            placeholder={conversationId === null ? 'Select a conversation first…' : 'Type a message…'}
            className="min-h-[3rem] flex-1 resize-none rounded border border-slate-700 bg-background px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-primary disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || conversationId === null || draft.trim().length === 0}
            className="rounded bg-primary px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-50"
          >
            {isLoading ? 'Sending…' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatPage;
