import axios, { AxiosError, type AxiosResponse } from 'axios';

import { useAuthStore } from '../store/authStore';

const BASE_URL = '/api/v1';

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

const extractMessage = (error: AxiosError): string => {
  const data = error.response?.data as { detail?: unknown } | undefined;
  if (typeof data?.detail === 'string') {
    return data.detail;
  }
  if (Array.isArray(data?.detail)) {
    const messages = data.detail
      .map((item) => (typeof item === 'object' && item !== null && 'msg' in item ? String((item as { msg: unknown }).msg) : ''))
      .filter((message) => message.length > 0);
    if (messages.length > 0) {
      return messages.join(', ');
    }
  }
  if (error.message) {
    return error.message;
  }
  return 'Something went wrong';
};

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => response,
  (error: AxiosError): Promise<never> => {
    const status = error.response?.status;
    if (status === 401) {
      useAuthStore.getState().clearToken();
    }
    return Promise.reject(new ApiError(status ?? 0, extractMessage(error)));
  },
);
