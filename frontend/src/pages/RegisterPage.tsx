import { useForm } from 'react-hook-form';
import { Link, useNavigate } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';

import { ROUTES } from '../routes';
import { authService } from '../services/auth.service';
import type { UserCreate } from '../types/auth';

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USERNAME_PATTERN = /^[A-Za-z0-9_]+$/;

const inputClass =
  'w-full rounded border border-slate-700 bg-background px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-primary';

const RegisterPage = () => {
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<UserCreate>({ defaultValues: { email: '', username: '', password: '' } });

  const registerMutation = useMutation({
    mutationFn: authService.register,
    onSuccess: () => {
      navigate(ROUTES.login, { replace: true, state: { registered: true } });
    },
  });

  const onSubmit = (values: UserCreate) => {
    registerMutation.mutate(values);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="w-full max-w-sm rounded-lg border border-slate-800 bg-surface p-6">
        <h1 className="text-xl font-semibold text-slate-100">Create account</h1>
        <p className="mt-2 text-sm text-slate-400">Start using MyLLM.pro.</p>

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
            <label htmlFor="username" className="block text-xs font-medium text-slate-400">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              className={`mt-1 ${inputClass}`}
              {...register('username', {
                required: 'Username is required',
                minLength: { value: 3, message: 'Username must be at least 3 characters' },
                maxLength: { value: 64, message: 'Username must be at most 64 characters' },
                pattern: { value: USERNAME_PATTERN, message: 'Only letters, numbers and underscores' },
              })}
            />
            {errors.username && <p className="mt-1 text-xs text-rose-400">{errors.username.message}</p>}
          </div>

          <div>
            <label htmlFor="password" className="block text-xs font-medium text-slate-400">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="new-password"
              className={`mt-1 ${inputClass}`}
              {...register('password', {
                required: 'Password is required',
                minLength: { value: 12, message: 'Password must be at least 12 characters' },
                maxLength: { value: 128, message: 'Password must be at most 128 characters' },
              })}
            />
            {errors.password && <p className="mt-1 text-xs text-rose-400">{errors.password.message}</p>}
          </div>

          {registerMutation.isError && (
            <div className="rounded border border-rose-800 bg-rose-950 px-3 py-2 text-xs text-rose-400" role="alert">
              {registerMutation.error instanceof Error ? registerMutation.error.message : 'Registration failed'}
            </div>
          )}

          {registerMutation.isSuccess && (
            <div
              className="rounded border border-emerald-800 bg-emerald-950 px-3 py-2 text-xs text-emerald-400"
              role="status"
            >
              Account created. Redirecting to sign in…
            </div>
          )}

          <button
            type="submit"
            disabled={registerMutation.isPending}
            className="w-full rounded bg-primary px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:opacity-50"
          >
            {registerMutation.isPending ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <p className="mt-4 text-center text-xs text-slate-400">
          Already registered?{' '}
          <Link to={ROUTES.login} className="text-primary hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
};

export default RegisterPage;
