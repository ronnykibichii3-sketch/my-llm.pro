import { create } from 'zustand';

export type SystemStatus = 'ONLINE' | 'OFFLINE';

interface SystemState {
  status: SystemStatus;
  setStatus: (status: SystemStatus) => void;
}

export const useSystemStore = create<SystemState>((set) => ({
  status: 'OFFLINE',
  setStatus: (status: SystemStatus) => set({ status }),
}));
