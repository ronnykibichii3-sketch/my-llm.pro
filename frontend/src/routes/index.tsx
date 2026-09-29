import { createBrowserRouter, Navigate } from 'react-router-dom';

import AppLayout from '../layout/AppLayout';
import AgentsPage from '../pages/AgentsPage';
import AutomationPage from '../pages/AutomationPage';
import ChatPage from '../pages/ChatPage';
import LoginPage from '../pages/LoginPage';
import MemoryPage from '../pages/MemoryPage';
import ModelsPage from '../pages/ModelsPage';
import RegisterPage from '../pages/RegisterPage';
import SettingsPage from '../pages/SettingsPage';
import ProtectedRoute from './ProtectedRoute';

export const ROUTES = {
  chat: '/',
  login: '/login',
  register: '/register',
  memory: '/memory',
  agents: '/agents',
  automation: '/automation',
  models: '/models',
  settings: '/settings',
} as const;

export const router = createBrowserRouter([
  { path: ROUTES.login, element: <LoginPage /> },
  { path: ROUTES.register, element: <RegisterPage /> },
  {
    element: (
      <ProtectedRoute>
        <AppLayout />
      </ProtectedRoute>
    ),
    children: [
      { path: ROUTES.chat, element: <ChatPage /> },
      { path: ROUTES.memory, element: <MemoryPage /> },
      { path: ROUTES.agents, element: <AgentsPage /> },
      { path: ROUTES.automation, element: <AutomationPage /> },
      { path: ROUTES.models, element: <ModelsPage /> },
      { path: ROUTES.settings, element: <SettingsPage /> },
      { path: '*', element: <Navigate to={ROUTES.chat} replace /> },
    ],
  },
]);
