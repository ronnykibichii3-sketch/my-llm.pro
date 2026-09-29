import { useForm } from 'react-hook-form';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';

import { ROUTES } from '../routes';
import { authService } from '../services/auth.service';
import { useAuthStore } from '../store/authStore';
import type { LoginRequest } from '../types/auth';

interface LocationState {
  from?: string;
}

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const inputClass =
  'w-full rounded border border-slate-700 bg-background px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-primary';

const LoginPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const token = useAuthStore((state) => state.token);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginRequest>({ defaultValues: { email: '', password: '' } });

  const loginMutation = useMutation({
    mutationFn: authService.login,
    onSuccess: () => {
      const state = location.state as LocationState | null;
      navigate(state?.from ?? ROUTES.chat, { replace: true });
    },
  });

  if (token) {
    return <Navigate to={ROUTES.chat} replace />;
  }

  const onSubmit = (values: LoginRequest) => {
    loginMutation.mutate(values);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="w-full max-w-sm rounded-lg border border-slate-800 bg-surface p-6">
        <h1 className="text-xl font-semibold text-slate-100">Sign in</h1>
        <p className="mt-2 text-sm text-slate-400">Access your AI operating system.</p>

        <form onSubmit={handleSubmit(onSubmit)} className="mt-6 space-y-4" noValidate>
          <div>
            <label htmlFor="email" className="block text-xs font-medium text-slate-400">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              className={`mt-1 ${inputClass}`}
              {...register('email', {
                required: 'Email is required',
                pattern: { value: EMAIL_PATTERN, message: 'Enter a valid email address' },
              })}
            />
            {errors.email && <p className="mt-1 text-xs text-rose-400">{errors.email.message}</p>}
          </div>

          <div>
            <label htmlFor="password" className="block text-xs font-medium text-slate-400">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              className={`mt-1 ${inputClass}`}
              {...register('password', { required: 'Password is required' })}
            />
            {errors.password && <p className="mt-1 text-xs text-rose-400">{errors.password.message}</p>}
          </div>

          {loginMutation.isError && (
            <div className="rounded border border-rose-800 bg-rose-950 px-3 py-2 text-xs text-rose-400" role="alert">
              {loginMutation.error instanceof Error ? loginMutation.error.message : 'Login failed'}
            </div>
          )}

          <button
            type="submit"
            disabled={loginMutation.isPending}
            className="w-full rounded bg-primary px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-50"
          >
            {loginMutation.isPending ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-slate-400">
          No account?{' '}
          <Link to={ROUTES.register} className="text-primary hover:underline">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
