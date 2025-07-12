import React from 'react';
import { Outlet, Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Layout: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
  };

  const isActive = (path: string) => {
    return location.pathname === path ? 'active' : '';
  };

  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <div className="header-content">
            <Link to="/dashboard" className="logo">
              SkillSwap
            </Link>
            
            <nav>
              <ul className="nav-links">
                <li>
                  <Link to="/dashboard" className={isActive('/dashboard')}>
                    Dashboard
                  </Link>
                </li>
                <li>
                  <Link to="/users" className={isActive('/users')}>
                    Browse Users
                  </Link>
                </li>
                <li>
                  <Link to="/skills" className={isActive('/skills')}>
                    Skills
                  </Link>
                </li>
                <li>
                  <Link to="/swaps" className={isActive('/swaps')}>
                    Swap Requests
                  </Link>
                </li>
                <li>
                  <Link to="/ratings" className={isActive('/ratings')}>
                    My Ratings
                  </Link>
                </li>
                {user?.is_admin && (
                  <li>
                    <Link to="/admin" className={isActive('/admin')}>
                      Admin
                    </Link>
                  </li>
                )}
              </ul>
            </nav>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                {user?.profile_photo && (
                  <img
                    src={user.profile_photo}
                    alt={user.name}
                    className="avatar avatar-small"
                  />
                )}
                <span className="font-medium">{user?.name}</span>
              </div>
              
              <div className="flex items-center gap-2">
                <Link
                  to="/profile/edit"
                  className="btn btn-secondary btn-small"
                >
                  Profile
                </Link>
                <button
                  onClick={handleLogout}
                  className="btn btn-outline btn-small"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Layout;