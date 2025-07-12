# Skill Swap Platform

A modern web application that enables users to list their skills and request others in return, facilitating skill exchanges within a community.

## Features

### User Features
- **User Registration & Authentication** - Secure user accounts with JWT authentication
- **Profile Management** - Complete user profiles with photos, bio, location, and availability
- **Skills Management** - Add offered skills with proficiency levels and wanted skills with urgency
- **User Discovery** - Browse and search users by skills, location, and other criteria
- **Swap Requests** - Create, accept, reject, and manage skill swap requests
- **Rating System** - Rate and provide feedback after completed swaps
- **Privacy Controls** - Make profiles public or private

### Admin Features
- **User Management** - Ban/unban users, view user statistics
- **Content Moderation** - Approve/reject skill descriptions
- **Platform Messages** - Send platform-wide announcements
- **Analytics & Reports** - Monitor user activity, swap statistics, and ratings
- **Data Export** - Export user data, swap logs, and feedback reports

## Technology Stack

### Backend
- **Node.js** with Express.js framework
- **SQLite** database for data storage
- **JWT** for authentication
- **Bcrypt** for password hashing
- **Multer** for file uploads
- **Express Validator** for input validation

### Frontend
- **React 18** with TypeScript
- **React Router** for navigation
- **React Query** for server state management
- **Styled Components** for styling
- **React Hook Form** for form management
- **React Toastify** for notifications

## Getting Started

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd skill-swap-platform
   ```

2. **Install dependencies**
   ```bash
   npm run install-all
   ```

3. **Set up the database**
   ```bash
   npm run setup
   ```

4. **Start the development servers**
   ```bash
   npm run dev
   ```

This will start:
- Backend server at `http://localhost:5000`
- Frontend development server at `http://localhost:3000`

### Environment Variables

Create a `.env` file in the server directory:

```env
NODE_ENV=development
PORT=5000
JWT_SECRET=your-jwt-secret-key
CLIENT_URL=http://localhost:3000
```

## Usage

### For Users

1. **Register an Account**
   - Create a new account with username, email, and password
   - Fill out your profile with skills, location, and availability

2. **Add Skills**
   - Add skills you can offer with proficiency levels
   - Add skills you want to learn with urgency levels

3. **Browse Users**
   - Search for users by skills, location, or other criteria
   - View user profiles and ratings

4. **Request Skill Swaps**
   - Send swap requests to other users
   - Specify what skill you want and what you're offering in return

5. **Manage Requests**
   - Accept or reject incoming requests
   - Track the status of your sent requests

6. **Complete Swaps**
   - Mark swaps as completed after the exchange
   - Rate and provide feedback on the experience

### For Admins

1. **Access Admin Panel**
   - Log in with admin credentials
   - Navigate to `/admin` to access the dashboard

2. **User Management**
   - View all users and their activity
   - Ban/unban users who violate policies
   - Monitor user statistics

3. **Content Moderation**
   - Review and approve new skill submissions
   - Remove inappropriate content

4. **Platform Management**
   - Send platform-wide messages
   - Generate activity reports
   - Export data for analysis

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/verify` - Verify JWT token

### Users
- `GET /api/users` - Get all users
- `GET /api/users/:id` - Get user profile
- `PUT /api/users/:id` - Update user profile
- `POST /api/users/:id/photo` - Upload profile photo

### Skills
- `GET /api/skills` - Get all skills
- `POST /api/skills` - Create new skill
- `GET /api/skills/search/users` - Search users by skill

### Swaps
- `GET /api/swaps/my` - Get user's swap requests
- `POST /api/swaps` - Create swap request
- `POST /api/swaps/:id/accept` - Accept swap request
- `POST /api/swaps/:id/reject` - Reject swap request

### Ratings
- `POST /api/ratings` - Create rating
- `GET /api/ratings/user/:userId` - Get user's ratings
- `GET /api/ratings/my/received` - Get current user's received ratings

### Admin
- `GET /api/admin/dashboard` - Get admin dashboard stats
- `GET /api/admin/users` - Get all users (admin)
- `POST /api/admin/users/:id/ban` - Ban/unban user
- `GET /api/admin/export/users` - Export user data

## Database Schema

### Users
- Profile information, authentication details
- Public/private settings, admin flags

### Skills
- Skill name, category, description
- Approval status for moderation

### User Skills
- Links users to offered/wanted skills
- Proficiency levels and urgency settings

### Swap Requests
- Request details, status tracking
- Links between users and skills

### Ratings
- User ratings and feedback
- Linked to completed swaps

### Admin Features
- Platform messages
- Admin action logs

## Security Features

- **JWT Authentication** - Secure token-based authentication
- **Password Hashing** - Bcrypt for secure password storage
- **Input Validation** - Server-side validation for all inputs
- **Rate Limiting** - API rate limiting to prevent abuse
- **CORS Configuration** - Cross-origin resource sharing controls
- **SQL Injection Prevention** - Parameterized queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

## Default Admin Account

- **Email:** admin@skillswap.com
- **Password:** admin123

⚠️ **Important:** Change the default admin password in production!

## Deployment

### Production Setup

1. **Build the frontend**
   ```bash
   cd client && npm run build
   ```

2. **Set production environment variables**
   ```bash
   NODE_ENV=production
   JWT_SECRET=your-secure-jwt-secret
   ```

3. **Run the production server**
   ```bash
   cd server && npm start
   ```

### Docker Support

Docker configuration can be added for containerized deployment.

## Roadmap

- [ ] Real-time notifications
- [ ] Mobile app (React Native)
- [ ] Video chat integration
- [ ] Skill verification system
- [ ] Advanced matching algorithms
- [ ] Social features (groups, events)
- [ ] Payment integration for premium features

---

Built with ❤️ for the skill-sharing community