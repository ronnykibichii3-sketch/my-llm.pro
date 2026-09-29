import type { ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

import { useAuthStore } from '../store/authStore';
import { ROUTES } from './index';

interface ProtectedRouteProps {
  children: ReactNode;
}

const ProtectedRoute = ({ children }: ProtectedRouteProps) => {
  const token = useAuthStore((state) => state.token);
  const location = useLocation();

  if (!token) {
    return <Navigate to={ROUTES.login} replace state={{ from: location.pathname }} />;
  }

  return children;
};

export default ProtectedRoute;
