import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const Dashboard: React.FC = () => {
  const { user } = useAuth();

  return (
    <div className="dashboard">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          Welcome back, {user?.name}!
        </h1>
        <p className="text-gray-600 mt-2">
          Ready to swap some skills? Here's your dashboard overview.
        </p>
      </div>

      <div className="grid grid-3 gap-6 mb-8">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">My Skills</h3>
            <Link to="/profile/edit" className="btn btn-small btn-outline">
              Manage
            </Link>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">0</div>
            <p className="text-sm text-gray-600">Skills Offered</p>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Active Swaps</h3>
            <Link to="/swaps" className="btn btn-small btn-outline">
              View All
            </Link>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-secondary">0</div>
            <p className="text-sm text-gray-600">Pending Requests</p>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">My Rating</h3>
            <Link to="/ratings" className="btn btn-small btn-outline">
              View Details
            </Link>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-success">0.0</div>
            <p className="text-sm text-gray-600">Average Rating</p>
          </div>
        </div>
      </div>

      <div className="grid grid-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Quick Actions</h3>
          </div>
          <div className="space-y-3">
            <Link to="/profile/edit" className="btn btn-primary btn-block">
              Complete Your Profile
            </Link>
            <Link to="/users" className="btn btn-secondary btn-block">
              Browse Users
            </Link>
            <Link to="/skills" className="btn btn-outline btn-block">
              Explore Skills
            </Link>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Recent Activity</h3>
          </div>
          <div className="text-center py-8">
            <p className="text-gray-500">No recent activity</p>
            <p className="text-sm text-gray-400 mt-2">
              Start by adding your skills or browsing other users!
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;