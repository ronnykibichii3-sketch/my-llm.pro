import { apiClient } from '../lib/api-client';
import { useAuthStore } from '../store/authStore';
import type { LoginRequest, TokenResponse, UserCreate, UserProfile, UserRead } from '../types/auth';

export const authService = {
  async register(payload: UserCreate): Promise<UserRead> {
    const { data } = await apiClient.post<UserRead>('/auth/register', payload);
    return data;
  },

  async login(payload: LoginRequest): Promise<void> {
    const { data } = await apiClient.post<TokenResponse>('/auth/login', payload);
    useAuthStore.getState().setToken(data.access_token);
    await authService.me();
  },

  async me(): Promise<UserProfile> {
    const { data } = await apiClient.get<UserProfile>('/auth/me');
    useAuthStore.getState().setUser(data);
    return data;
  },

  logout(): void {
    useAuthStore.getState().clearToken();
  },
};
