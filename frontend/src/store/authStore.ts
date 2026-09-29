import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type { UserProfile } from '../types/auth';

interface AuthState {
  token: string | null;
  user: UserProfile | null;
  setToken: (token: string) => void;
  setUser: (user: UserProfile) => void;
  clearToken: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      user: null,
      setToken: (token: string) => set({ token }),
      setUser: (user: UserProfile) => set({ user }),
      clearToken: () => set({ token: null, user: null }),
    }),
    { name: 'myllm-auth' },
  ),
);
