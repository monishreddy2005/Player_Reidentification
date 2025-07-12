const express = require('express');
const { body, validationResult } = require('express-validator');
const db = require('../config/database');
const { authenticateToken, requireAdmin } = require('../middleware/auth');

const router = express.Router();

// Apply admin middleware to all routes
router.use(authenticateToken);
router.use(requireAdmin);

// Dashboard stats
router.get('/dashboard', async (req, res) => {
  try {
    const stats = await Promise.all([
      // User stats
      db.get('SELECT COUNT(*) as total_users FROM users'),
      db.get('SELECT COUNT(*) as active_users FROM users WHERE is_banned = 0'),
      db.get('SELECT COUNT(*) as banned_users FROM users WHERE is_banned = 1'),
      
      // Skill stats
      db.get('SELECT COUNT(*) as total_skills FROM skills'),
      db.get('SELECT COUNT(*) as approved_skills FROM skills WHERE is_approved = 1'),
      db.get('SELECT COUNT(*) as pending_skills FROM skills WHERE is_approved = 0'),
      
      // Swap stats
      db.get('SELECT COUNT(*) as total_swaps FROM swap_requests'),
      db.get('SELECT COUNT(*) as pending_swaps FROM swap_requests WHERE status = "pending"'),
      db.get('SELECT COUNT(*) as completed_swaps FROM swap_requests WHERE status = "completed"'),
      
      // Rating stats
      db.get('SELECT COUNT(*) as total_ratings, AVG(rating) as avg_rating FROM ratings'),
      
      // Recent activity
      db.query(`
        SELECT 'user' as type, name as title, created_at 
        FROM users 
        WHERE created_at > datetime('now', '-7 days')
        UNION ALL
        SELECT 'swap' as type, 'New swap request' as title, created_at 
        FROM swap_requests 
        WHERE created_at > datetime('now', '-7 days')
        UNION ALL
        SELECT 'rating' as type, 'New rating' as title, created_at 
        FROM ratings 
        WHERE created_at > datetime('now', '-7 days')
        ORDER BY created_at DESC
        LIMIT 10
      `)
    ]);

    res.json({
      users: {
        total: stats[0].total_users,
        active: stats[1].active_users,
        banned: stats[2].banned_users
      },
      skills: {
        total: stats[3].total_skills,
        approved: stats[4].approved_skills,
        pending: stats[5].pending_skills
      },
      swaps: {
        total: stats[6].total_swaps,
        pending: stats[7].pending_swaps,
        completed: stats[8].completed_swaps
      },
      ratings: {
        total: stats[9].total_ratings,
        average: stats[9].avg_rating
      },
      recentActivity: stats[10]
    });
  } catch (error) {
    console.error('Dashboard stats error:', error);
    res.status(500).json({ error: 'Server error while fetching dashboard stats' });
  }
});

// User management
router.get('/users', async (req, res) => {
  try {
    const { search, status, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;

    let query = `
      SELECT u.*, 
             (SELECT COUNT(*) FROM user_offered_skills WHERE user_id = u.id) as skills_offered,
             (SELECT COUNT(*) FROM user_wanted_skills WHERE user_id = u.id) as skills_wanted,
             (SELECT COUNT(*) FROM swap_requests WHERE requester_id = u.id OR provider_id = u.id) as total_swaps,
             (SELECT AVG(rating) FROM ratings WHERE rated_id = u.id) as avg_rating
      FROM users u
      WHERE 1=1
    `;

    const params = [];

    if (search) {
      query += ' AND (u.name LIKE ? OR u.username LIKE ? OR u.email LIKE ?)';
      const searchPattern = `%${search}%`;
      params.push(searchPattern, searchPattern, searchPattern);
    }

    if (status === 'banned') {
      query += ' AND u.is_banned = 1';
    } else if (status === 'active') {
      query += ' AND u.is_banned = 0';
    }

    query += ' ORDER BY u.created_at DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);

    const users = await db.query(query, params);

    res.json({ users, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get users error:', error);
    res.status(500).json({ error: 'Server error while fetching users' });
  }
});

// Ban/unban user
router.post('/users/:id/ban', async (req, res) => {
  try {
    const userId = req.params.id;
    const { reason } = req.body;

    const user = await db.get('SELECT * FROM users WHERE id = ?', [userId]);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (user.is_admin) {
      return res.status(400).json({ error: 'Cannot ban admin users' });
    }

    const newBanStatus = user.is_banned ? 0 : 1;
    
    await db.run('UPDATE users SET is_banned = ? WHERE id = ?', [newBanStatus, userId]);

    // Log admin action
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, newBanStatus ? 'ban_user' : 'unban_user', 'user', userId, reason || 'No reason provided']
    );

    res.json({ message: `User ${newBanStatus ? 'banned' : 'unbanned'} successfully` });
  } catch (error) {
    console.error('Ban/unban user error:', error);
    res.status(500).json({ error: 'Server error while updating user ban status' });
  }
});

// Delete user
router.delete('/users/:id', async (req, res) => {
  try {
    const userId = req.params.id;
    const { reason } = req.body;

    const user = await db.get('SELECT * FROM users WHERE id = ?', [userId]);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (user.is_admin) {
      return res.status(400).json({ error: 'Cannot delete admin users' });
    }

    // Log admin action before deletion
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, 'delete_user', 'user', userId, reason || 'No reason provided']
    );

    // Delete user (cascade will handle related records)
    await db.run('DELETE FROM users WHERE id = ?', [userId]);

    res.json({ message: 'User deleted successfully' });
  } catch (error) {
    console.error('Delete user error:', error);
    res.status(500).json({ error: 'Server error while deleting user' });
  }
});

// Skill management
router.get('/skills', async (req, res) => {
  try {
    const { status, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;

    let query = `
      SELECT s.*, 
             (SELECT COUNT(*) FROM user_offered_skills WHERE skill_id = s.id) as offered_count,
             (SELECT COUNT(*) FROM user_wanted_skills WHERE skill_id = s.id) as wanted_count
      FROM skills s
      WHERE 1=1
    `;

    const params = [];

    if (status === 'pending') {
      query += ' AND s.is_approved = 0';
    } else if (status === 'approved') {
      query += ' AND s.is_approved = 1';
    }

    query += ' ORDER BY s.created_at DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);

    const skills = await db.query(query, params);

    res.json({ skills, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get skills error:', error);
    res.status(500).json({ error: 'Server error while fetching skills' });
  }
});

// Approve/reject skill
router.post('/skills/:id/approve', async (req, res) => {
  try {
    const skillId = req.params.id;
    const { approved } = req.body;

    const skill = await db.get('SELECT * FROM skills WHERE id = ?', [skillId]);
    if (!skill) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    await db.run('UPDATE skills SET is_approved = ? WHERE id = ?', [approved ? 1 : 0, skillId]);

    // Log admin action
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, approved ? 'approve_skill' : 'reject_skill', 'skill', skillId, `Skill: ${skill.name}`]
    );

    res.json({ message: `Skill ${approved ? 'approved' : 'rejected'} successfully` });
  } catch (error) {
    console.error('Approve/reject skill error:', error);
    res.status(500).json({ error: 'Server error while updating skill approval status' });
  }
});

// Delete skill
router.delete('/skills/:id', async (req, res) => {
  try {
    const skillId = req.params.id;
    const { reason } = req.body;

    const skill = await db.get('SELECT * FROM skills WHERE id = ?', [skillId]);
    if (!skill) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    // Check if skill is in use
    const inUse = await db.get(`
      SELECT COUNT(*) as count FROM (
        SELECT user_id FROM user_offered_skills WHERE skill_id = ?
        UNION ALL
        SELECT user_id FROM user_wanted_skills WHERE skill_id = ?
      )
    `, [skillId, skillId]);

    if (inUse.count > 0) {
      return res.status(400).json({ error: 'Cannot delete skill that is in use by users' });
    }

    // Log admin action before deletion
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, 'delete_skill', 'skill', skillId, reason || `Deleted skill: ${skill.name}`]
    );

    await db.run('DELETE FROM skills WHERE id = ?', [skillId]);

    res.json({ message: 'Skill deleted successfully' });
  } catch (error) {
    console.error('Delete skill error:', error);
    res.status(500).json({ error: 'Server error while deleting skill' });
  }
});

// Platform messages
router.get('/messages', async (req, res) => {
  try {
    const { page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;

    const messages = await db.query(`
      SELECT am.*, u.name as admin_name, u.username as admin_username
      FROM admin_messages am
      JOIN users u ON am.admin_id = u.id
      ORDER BY am.created_at DESC
      LIMIT ? OFFSET ?
    `, [parseInt(limit), offset]);

    res.json({ messages, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get messages error:', error);
    res.status(500).json({ error: 'Server error while fetching messages' });
  }
});

// Create platform message
router.post('/messages', [
  body('title').isLength({ min: 1, max: 200 }).trim().escape(),
  body('content').isLength({ min: 1, max: 1000 }).trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { title, content } = req.body;

    const result = await db.run(
      'INSERT INTO admin_messages (admin_id, title, content) VALUES (?, ?, ?)',
      [req.user.id, title, content]
    );

    // Log admin action
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, 'create_message', 'message', result.lastID, `Message: ${title}`]
    );

    res.status(201).json({ message: 'Platform message created successfully', id: result.lastID });
  } catch (error) {
    console.error('Create message error:', error);
    res.status(500).json({ error: 'Server error while creating message' });
  }
});

// Update platform message
router.put('/messages/:id', [
  body('title').optional().isLength({ min: 1, max: 200 }).trim().escape(),
  body('content').optional().isLength({ min: 1, max: 1000 }).trim().escape(),
  body('is_active').optional().isBoolean()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const messageId = req.params.id;
    const { title, content, is_active } = req.body;

    const updates = [];
    const params = [];

    if (title !== undefined) {
      updates.push('title = ?');
      params.push(title);
    }
    if (content !== undefined) {
      updates.push('content = ?');
      params.push(content);
    }
    if (is_active !== undefined) {
      updates.push('is_active = ?');
      params.push(is_active ? 1 : 0);
    }

    if (updates.length === 0) {
      return res.status(400).json({ error: 'No valid fields to update' });
    }

    params.push(messageId);

    const result = await db.run(
      `UPDATE admin_messages SET ${updates.join(', ')} WHERE id = ?`,
      params
    );

    if (result.changes === 0) {
      return res.status(404).json({ error: 'Message not found' });
    }

    // Log admin action
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, 'update_message', 'message', messageId, 'Updated platform message']
    );

    res.json({ message: 'Platform message updated successfully' });
  } catch (error) {
    console.error('Update message error:', error);
    res.status(500).json({ error: 'Server error while updating message' });
  }
});

// Delete platform message
router.delete('/messages/:id', async (req, res) => {
  try {
    const messageId = req.params.id;

    const result = await db.run('DELETE FROM admin_messages WHERE id = ?', [messageId]);

    if (result.changes === 0) {
      return res.status(404).json({ error: 'Message not found' });
    }

    // Log admin action
    await db.run(
      'INSERT INTO admin_actions (admin_id, action_type, target_type, target_id, description) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, 'delete_message', 'message', messageId, 'Deleted platform message']
    );

    res.json({ message: 'Platform message deleted successfully' });
  } catch (error) {
    console.error('Delete message error:', error);
    res.status(500).json({ error: 'Server error while deleting message' });
  }
});

// Get active platform messages (for public display)
router.get('/messages/active', async (req, res) => {
  try {
    const messages = await db.query(`
      SELECT am.id, am.title, am.content, am.created_at
      FROM admin_messages am
      WHERE am.is_active = 1
      ORDER BY am.created_at DESC
      LIMIT 5
    `);

    res.json({ messages });
  } catch (error) {
    console.error('Get active messages error:', error);
    res.status(500).json({ error: 'Server error while fetching active messages' });
  }
});

// Admin actions log
router.get('/actions', async (req, res) => {
  try {
    const { page = 1, limit = 50 } = req.query;
    const offset = (page - 1) * limit;

    const actions = await db.query(`
      SELECT aa.*, u.name as admin_name, u.username as admin_username
      FROM admin_actions aa
      JOIN users u ON aa.admin_id = u.id
      ORDER BY aa.created_at DESC
      LIMIT ? OFFSET ?
    `, [parseInt(limit), offset]);

    res.json({ actions, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get admin actions error:', error);
    res.status(500).json({ error: 'Server error while fetching admin actions' });
  }
});

// Generate reports
router.get('/reports/user-activity', async (req, res) => {
  try {
    const { startDate, endDate } = req.query;
    
    let dateFilter = '';
    const params = [];
    
    if (startDate && endDate) {
      dateFilter = 'WHERE created_at BETWEEN ? AND ?';
      params.push(startDate, endDate);
    } else if (startDate) {
      dateFilter = 'WHERE created_at >= ?';
      params.push(startDate);
    } else if (endDate) {
      dateFilter = 'WHERE created_at <= ?';
      params.push(endDate);
    }

    const [userStats, swapStats, ratingStats] = await Promise.all([
      db.query(`
        SELECT 
          DATE(created_at) as date,
          COUNT(*) as new_users
        FROM users 
        ${dateFilter}
        GROUP BY DATE(created_at)
        ORDER BY date DESC
      `, params),
      
      db.query(`
        SELECT 
          DATE(created_at) as date,
          COUNT(*) as new_swaps,
          COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_swaps
        FROM swap_requests 
        ${dateFilter}
        GROUP BY DATE(created_at)
        ORDER BY date DESC
      `, params),
      
      db.query(`
        SELECT 
          DATE(created_at) as date,
          COUNT(*) as new_ratings,
          AVG(rating) as avg_rating
        FROM ratings 
        ${dateFilter}
        GROUP BY DATE(created_at)
        ORDER BY date DESC
      `, params)
    ]);

    res.json({
      userStats,
      swapStats,
      ratingStats
    });
  } catch (error) {
    console.error('Generate report error:', error);
    res.status(500).json({ error: 'Server error while generating report' });
  }
});

// Export data
router.get('/export/users', async (req, res) => {
  try {
    const users = await db.query(`
      SELECT id, username, email, name, location, created_at, is_public, is_banned
      FROM users
      ORDER BY created_at DESC
    `);

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename=users-export.json');
    res.json(users);
  } catch (error) {
    console.error('Export users error:', error);
    res.status(500).json({ error: 'Server error while exporting users' });
  }
});

router.get('/export/swaps', async (req, res) => {
  try {
    const swaps = await db.query(`
      SELECT 
        sr.id, sr.status, sr.created_at, sr.updated_at,
        requester.username as requester_username,
        provider.username as provider_username,
        rs.name as requested_skill,
        os.name as offered_skill
      FROM swap_requests sr
      JOIN users requester ON sr.requester_id = requester.id
      JOIN users provider ON sr.provider_id = provider.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      ORDER BY sr.created_at DESC
    `);

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename=swaps-export.json');
    res.json(swaps);
  } catch (error) {
    console.error('Export swaps error:', error);
    res.status(500).json({ error: 'Server error while exporting swaps' });
  }
});

router.get('/export/ratings', async (req, res) => {
  try {
    const ratings = await db.query(`
      SELECT 
        r.id, r.rating, r.feedback, r.created_at,
        rater.username as rater_username,
        rated.username as rated_username
      FROM ratings r
      JOIN users rater ON r.rater_id = rater.id
      JOIN users rated ON r.rated_id = rated.id
      ORDER BY r.created_at DESC
    `);

    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', 'attachment; filename=ratings-export.json');
    res.json(ratings);
  } catch (error) {
    console.error('Export ratings error:', error);
    res.status(500).json({ error: 'Server error while exporting ratings' });
  }
});

module.exports = router;