import { useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';

import { ROUTES } from '../routes';
import { authService } from '../services/auth.service';
import { HEALTH_POLL_INTERVAL_MS, systemService } from '../services/system.service';
import { useAuthStore } from '../store/authStore';
import { useSystemStore, type SystemStatus } from '../store/systemStore';

const resolveStatus = (isSuccess: boolean, isHealthOk: boolean): SystemStatus => {
  if (isSuccess && isHealthOk) return 'ONLINE';
  return 'OFFLINE';
};

const Topbar = () => {
  const navigate = useNavigate();
  const status = useSystemStore((state) => state.status);
  const setStatus = useSystemStore((state) => state.setStatus);
  const token = useAuthStore((state) => state.token);
  const user = useAuthStore((state) => state.user);

  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: systemService.getHealth,
    refetchInterval: HEALTH_POLL_INTERVAL_MS,
  });

  useQuery({
    queryKey: ['me'],
    queryFn: authService.me,
    enabled: Boolean(token) && !user,
    staleTime: 60_000,
  });

  const logoutMutation = useMutation({
    mutationFn: async () => authService.logout(),
    onSuccess: () => navigate(ROUTES.login, { replace: true }),
  });

  const isHealthOk = healthQuery.data?.status === 'ok';

  useEffect(() => {
    setStatus(resolveStatus(healthQuery.isSuccess, isHealthOk));
  }, [healthQuery.isSuccess, isHealthOk, setStatus]);

  const isOnline = status === 'ONLINE';

  return (
    <header className="flex h-14 items-center justify-between border-b border-slate-800 bg-surface px-4">
      <span className="text-sm text-slate-400">AI Operating System</span>
      <div className="flex items-center gap-3">
        {user && (
          <span className="text-sm text-slate-300" title={user.email}>
            {user.username}
          </span>
        )}
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${
            isOnline ? 'border-emerald-700 bg-emerald-950 text-emerald-400' : 'border-rose-800 bg-rose-950 text-rose-400'
          }`}
        >
          <span className={`h-2 w-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-rose-500'}`} />
          {status}
        </span>
        <button
          type="button"
          onClick={() => logoutMutation.mutate()}
          disabled={logoutMutation.isPending}
          className="rounded border border-slate-700 px-3 py-1 text-xs text-slate-300 transition hover:bg-slate-800 disabled:opacity-50"
        >
          {logoutMutation.isPending ? 'Signing out…' : 'Sign out'}
        </button>
      </div>
    </header>
  );
};

export default Topbar;
