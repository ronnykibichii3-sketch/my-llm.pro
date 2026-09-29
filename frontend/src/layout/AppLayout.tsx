import { Outlet } from 'react-router-dom';

import Sidebar from './Sidebar';
import Topbar from './Topbar';

const AppLayout = () => {
  return (
    <div className="flex h-screen flex-col bg-background text-slate-100">
      <Topbar />
      <div className="flex min-h-0 flex-1">
        <Sidebar />
        <main className="min-w-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default AppLayout;
