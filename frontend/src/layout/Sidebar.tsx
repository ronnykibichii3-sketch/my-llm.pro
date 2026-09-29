import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { NavLink, useNavigate } from 'react-router-dom';

import { ROUTES } from '../routes';
import { conversationService } from '../services/conversation.service';
import { useChatStore } from '../store/chatStore';

const NAV_ITEMS = [
  { to: ROUTES.chat, label: 'Chat', end: true },
  { to: ROUTES.memory, label: 'Memory', end: false },
  { to: ROUTES.agents, label: 'Agents', end: false },
  { to: ROUTES.automation, label: 'Automation', end: false },
  { to: ROUTES.models, label: 'Models', end: false },
  { to: ROUTES.settings, label: 'Settings', end: false },
] as const;

const NEW_CONVERSATION_TITLE = 'New conversation';

const Sidebar = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const activeConversationId = useChatStore((state) => state.conversationId);
  const setConversationId = useChatStore((state) => state.setConversationId);

  const conversationsQuery = useQuery({
    queryKey: ['conversations'],
    queryFn: conversationService.list,
  });

  const conversationCount = conversationsQuery.data?.length ?? 0;

  const createConversationMutation = useMutation({
    mutationFn: conversationService.create,
    onSuccess: (conversation) => {
      void queryClient.invalidateQueries({ queryKey: ['conversations'] });
      setConversationId(conversation.id);
      navigate(ROUTES.chat);
    },
  });

  const handleNewConversation = () => {
    createConversationMutation.mutate({ title: NEW_CONVERSATION_TITLE });
  };

  const handleSelectConversation = (id: number) => {
    setConversationId(id);
    navigate(ROUTES.chat);
  };

  return (
    <aside className="flex h-full w-64 flex-col border-r border-slate-800 bg-surface">
      <div className="flex h-14 items-center border-b border-slate-800 px-4">
        <span className="text-lg font-semibold text-primary">MyLLM.pro</span>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto p-3">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `block rounded px-3 py-2 text-sm transition ${
                isActive ? 'bg-primary/15 text-primary' : 'text-slate-300 hover:bg-slate-800'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}

        <div className="border-t border-slate-800 pt-3">
          <div className="flex items-center justify-between px-3">
            <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Conversations
              {conversationsQuery.isSuccess && (
                <span className="ml-1.5 font-normal text-slate-600">({conversationCount})</span>
              )}
            </span>
            <button
              type="button"
              onClick={handleNewConversation}
              disabled={createConversationMutation.isPending}
              className="rounded bg-primary px-2 py-0.5 text-xs font-medium text-white transition hover:opacity-90 disabled:opacity-50"
            >
              {createConversationMutation.isPending ? 'Creating…' : '+ New'}
            </button>
          </div>

          {createConversationMutation.isError && (
            <p className="mt-2 px-3 text-xs text-rose-400">
              {createConversationMutation.error instanceof Error
                ? createConversationMutation.error.message
                : 'Failed to create conversation'}
            </p>
          )}

          {conversationsQuery.isPending && <p className="mt-2 px-3 text-xs text-slate-500">Loading conversations…</p>}

          {conversationsQuery.isError && (
            <div className="mt-2 px-3 text-xs text-rose-400">
              <p>{conversationsQuery.error instanceof Error ? conversationsQuery.error.message : 'Failed to load'}</p>
              <button
                type="button"
                onClick={() => void conversationsQuery.refetch()}
                className="mt-1 underline hover:text-rose-300"
              >
                Retry
              </button>
            </div>
          )}

          {conversationsQuery.isSuccess && conversationsQuery.data.length === 0 && (
            <p className="mt-2 px-3 text-xs text-slate-500">No conversations yet. Create one to get started.</p>
          )}

          {conversationsQuery.isSuccess && conversationsQuery.data.length > 0 && (
            <ul className="mt-2 space-y-1">
              {conversationsQuery.data.map((conversation) => {
                const isSelected = conversation.id === activeConversationId;
                return (
                  <li key={conversation.id}>
                    <button
                      type="button"
                      onClick={() => handleSelectConversation(conversation.id)}
                      aria-current={isSelected ? 'true' : undefined}
                      className={`block w-full truncate rounded px-3 py-1.5 text-left text-sm transition ${
                        isSelected ? 'bg-primary/15 text-primary' : 'text-slate-300 hover:bg-slate-800'
                      }`}
                      title={conversation.title}
                    >
                      {conversation.title}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </nav>
    </aside>
  );
};

export default Sidebar;
