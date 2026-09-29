import { apiClient } from '../lib/api-client';

export interface HealthResponse {
  status: string;
}

export const HEALTH_POLL_INTERVAL_MS = 15_000;

export const systemService = {
  async getHealth(): Promise<HealthResponse> {
    const { data } = await apiClient.get<HealthResponse>('/health');
    return data;
  },
};
