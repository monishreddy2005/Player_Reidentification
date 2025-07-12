import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import Layout from './components/Layout';
import LoadingSpinner from './components/LoadingSpinner';

// Auth components
import Login from './components/auth/Login';
import Register from './components/auth/Register';

// Main components
import Dashboard from './components/Dashboard';
import UserProfile from './components/profile/UserProfile';
import EditProfile from './components/profile/EditProfile';
import BrowseUsers from './components/users/BrowseUsers';
import SkillsPage from './components/skills/SkillsPage';
import SwapRequests from './components/swaps/SwapRequests';
import MyRatings from './components/ratings/MyRatings';

// Admin components
import AdminDashboard from './components/admin/AdminDashboard';
import AdminUsers from './components/admin/AdminUsers';
import AdminSkills from './components/admin/AdminSkills';
import AdminSwaps from './components/admin/AdminSwaps';
import AdminMessages from './components/admin/AdminMessages';

// Protected route wrapper
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Admin route wrapper
const AdminRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (!user || !user.is_admin) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

// Public route wrapper (redirect to dashboard if already logged in)
const PublicRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return <LoadingSpinner />;
  }

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

function App() {
  return (
    <div className="App">
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        } />
        <Route path="/register" element={
          <PublicRoute>
            <Register />
          </PublicRoute>
        } />

        {/* Protected routes */}
        <Route path="/" element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="profile/:id" element={<UserProfile />} />
          <Route path="profile/edit" element={<EditProfile />} />
          <Route path="users" element={<BrowseUsers />} />
          <Route path="skills" element={<SkillsPage />} />
          <Route path="swaps" element={<SwapRequests />} />
          <Route path="ratings" element={<MyRatings />} />

          {/* Admin routes */}
          <Route path="admin" element={
            <AdminRoute>
              <AdminDashboard />
            </AdminRoute>
          } />
          <Route path="admin/users" element={
            <AdminRoute>
              <AdminUsers />
            </AdminRoute>
          } />
          <Route path="admin/skills" element={
            <AdminRoute>
              <AdminSkills />
            </AdminRoute>
          } />
          <Route path="admin/swaps" element={
            <AdminRoute>
              <AdminSwaps />
            </AdminRoute>
          } />
          <Route path="admin/messages" element={
            <AdminRoute>
              <AdminMessages />
            </AdminRoute>
          } />
        </Route>

        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </div>
  );
}

export default App;