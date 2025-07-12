const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { body, validationResult } = require('express-validator');
const db = require('../config/database');
const { authenticateToken, requireOwnershipOrAdmin } = require('../middleware/auth');

const router = express.Router();

// Configure multer for profile photo uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '../uploads/profiles');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'profile-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: function (req, file, cb) {
    const allowedTypes = /jpeg|jpg|png|gif/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

// Get all users (public profiles only, unless admin)
router.get('/', authenticateToken, async (req, res) => {
  try {
    const { search, skill, location, page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    let query = `
      SELECT DISTINCT u.id, u.username, u.name, u.location, u.profile_photo, u.bio, u.availability, u.created_at
      FROM users u
      LEFT JOIN user_offered_skills uos ON u.id = uos.user_id
      LEFT JOIN skills s ON uos.skill_id = s.id
      WHERE u.is_banned = 0 AND (u.is_public = 1 OR u.id = ? OR ? = 1)
    `;
    
    const params = [req.user.id, req.user.is_admin ? 1 : 0];

    if (search) {
      query += ` AND (u.name LIKE ? OR u.username LIKE ? OR u.bio LIKE ?)`;
      const searchPattern = `%${search}%`;
      params.push(searchPattern, searchPattern, searchPattern);
    }

    if (skill) {
      query += ` AND s.name LIKE ?`;
      params.push(`%${skill}%`);
    }

    if (location) {
      query += ` AND u.location LIKE ?`;
      params.push(`%${location}%`);
    }

    query += ` ORDER BY u.created_at DESC LIMIT ? OFFSET ?`;
    params.push(parseInt(limit), offset);

    const users = await db.query(query, params);

    // Get skills for each user
    for (let user of users) {
      const offeredSkills = await db.query(`
        SELECT s.id, s.name, s.category, uos.proficiency_level, uos.description
        FROM user_offered_skills uos
        JOIN skills s ON uos.skill_id = s.id
        WHERE uos.user_id = ?
      `, [user.id]);

      user.offeredSkills = offeredSkills;
    }

    res.json({ users, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get users error:', error);
    res.status(500).json({ error: 'Server error while fetching users' });
  }
});

// Get user profile
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const userId = req.params.id;

    // Check if user can view this profile
    const user = await db.get('SELECT * FROM users WHERE id = ?', [userId]);
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    if (!user.is_public && user.id !== req.user.id && !req.user.is_admin) {
      return res.status(403).json({ error: 'Profile is private' });
    }

    // Remove sensitive data
    const { password_hash, ...userProfile } = user;

    // Get offered skills
    const offeredSkills = await db.query(`
      SELECT s.id, s.name, s.category, uos.proficiency_level, uos.description
      FROM user_offered_skills uos
      JOIN skills s ON uos.skill_id = s.id
      WHERE uos.user_id = ?
    `, [userId]);

    // Get wanted skills
    const wantedSkills = await db.query(`
      SELECT s.id, s.name, s.category, uws.urgency, uws.description
      FROM user_wanted_skills uws
      JOIN skills s ON uws.skill_id = s.id
      WHERE uws.user_id = ?
    `, [userId]);

    // Get ratings summary
    const ratingSummary = await db.get(`
      SELECT AVG(rating) as average_rating, COUNT(*) as total_ratings
      FROM ratings
      WHERE rated_id = ?
    `, [userId]);

    userProfile.offeredSkills = offeredSkills;
    userProfile.wantedSkills = wantedSkills;
    userProfile.ratingSummary = ratingSummary;

    res.json(userProfile);
  } catch (error) {
    console.error('Get user profile error:', error);
    res.status(500).json({ error: 'Server error while fetching user profile' });
  }
});

// Update user profile
router.put('/:id', [
  authenticateToken,
  requireOwnershipOrAdmin('id'),
  body('name').optional().isLength({ min: 2 }).trim().escape(),
  body('location').optional().trim().escape(),
  body('bio').optional().trim().escape(),
  body('availability').optional().trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const userId = req.params.id;
    const { name, location, bio, availability, is_public } = req.body;

    const updates = [];
    const params = [];

    if (name !== undefined) {
      updates.push('name = ?');
      params.push(name);
    }
    if (location !== undefined) {
      updates.push('location = ?');
      params.push(location);
    }
    if (bio !== undefined) {
      updates.push('bio = ?');
      params.push(bio);
    }
    if (availability !== undefined) {
      updates.push('availability = ?');
      params.push(availability);
    }
    if (is_public !== undefined) {
      updates.push('is_public = ?');
      params.push(is_public ? 1 : 0);
    }

    if (updates.length === 0) {
      return res.status(400).json({ error: 'No valid fields to update' });
    }

    updates.push('updated_at = CURRENT_TIMESTAMP');
    params.push(userId);

    await db.run(
      `UPDATE users SET ${updates.join(', ')} WHERE id = ?`,
      params
    );

    res.json({ message: 'Profile updated successfully' });
  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({ error: 'Server error while updating profile' });
  }
});

// Upload profile photo
router.post('/:id/photo', authenticateToken, requireOwnershipOrAdmin('id'), upload.single('photo'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const userId = req.params.id;
    const photoPath = `/uploads/profiles/${req.file.filename}`;

    // Delete old photo if exists
    const oldUser = await db.get('SELECT profile_photo FROM users WHERE id = ?', [userId]);
    if (oldUser && oldUser.profile_photo) {
      const oldPhotoPath = path.join(__dirname, '../', oldUser.profile_photo);
      if (fs.existsSync(oldPhotoPath)) {
        fs.unlinkSync(oldPhotoPath);
      }
    }

    // Update user with new photo path
    await db.run(
      'UPDATE users SET profile_photo = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
      [photoPath, userId]
    );

    res.json({ 
      message: 'Photo uploaded successfully',
      photoPath: photoPath
    });
  } catch (error) {
    console.error('Upload photo error:', error);
    res.status(500).json({ error: 'Server error while uploading photo' });
  }
});

// Add offered skill
router.post('/:id/skills/offered', [
  authenticateToken,
  requireOwnershipOrAdmin('id'),
  body('skill_id').isInt(),
  body('proficiency_level').optional().isIn(['beginner', 'intermediate', 'advanced', 'expert']),
  body('description').optional().trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const userId = req.params.id;
    const { skill_id, proficiency_level = 'intermediate', description } = req.body;

    // Check if skill exists
    const skill = await db.get('SELECT id FROM skills WHERE id = ?', [skill_id]);
    if (!skill) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    // Check if user already offers this skill
    const existingSkill = await db.get(
      'SELECT id FROM user_offered_skills WHERE user_id = ? AND skill_id = ?',
      [userId, skill_id]
    );

    if (existingSkill) {
      return res.status(409).json({ error: 'Skill already offered by user' });
    }

    await db.run(
      'INSERT INTO user_offered_skills (user_id, skill_id, proficiency_level, description) VALUES (?, ?, ?, ?)',
      [userId, skill_id, proficiency_level, description]
    );

    res.status(201).json({ message: 'Skill added successfully' });
  } catch (error) {
    console.error('Add offered skill error:', error);
    res.status(500).json({ error: 'Server error while adding skill' });
  }
});

// Add wanted skill
router.post('/:id/skills/wanted', [
  authenticateToken,
  requireOwnershipOrAdmin('id'),
  body('skill_id').isInt(),
  body('urgency').optional().isIn(['low', 'medium', 'high']),
  body('description').optional().trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const userId = req.params.id;
    const { skill_id, urgency = 'medium', description } = req.body;

    // Check if skill exists
    const skill = await db.get('SELECT id FROM skills WHERE id = ?', [skill_id]);
    if (!skill) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    // Check if user already wants this skill
    const existingSkill = await db.get(
      'SELECT id FROM user_wanted_skills WHERE user_id = ? AND skill_id = ?',
      [userId, skill_id]
    );

    if (existingSkill) {
      return res.status(409).json({ error: 'Skill already wanted by user' });
    }

    await db.run(
      'INSERT INTO user_wanted_skills (user_id, skill_id, urgency, description) VALUES (?, ?, ?, ?)',
      [userId, skill_id, urgency, description]
    );

    res.status(201).json({ message: 'Wanted skill added successfully' });
  } catch (error) {
    console.error('Add wanted skill error:', error);
    res.status(500).json({ error: 'Server error while adding wanted skill' });
  }
});

// Remove offered skill
router.delete('/:id/skills/offered/:skillId', authenticateToken, requireOwnershipOrAdmin('id'), async (req, res) => {
  try {
    const userId = req.params.id;
    const skillId = req.params.skillId;

    const result = await db.run(
      'DELETE FROM user_offered_skills WHERE user_id = ? AND skill_id = ?',
      [userId, skillId]
    );

    if (result.changes === 0) {
      return res.status(404).json({ error: 'Offered skill not found' });
    }

    res.json({ message: 'Offered skill removed successfully' });
  } catch (error) {
    console.error('Remove offered skill error:', error);
    res.status(500).json({ error: 'Server error while removing offered skill' });
  }
});

// Remove wanted skill
router.delete('/:id/skills/wanted/:skillId', authenticateToken, requireOwnershipOrAdmin('id'), async (req, res) => {
  try {
    const userId = req.params.id;
    const skillId = req.params.skillId;

    const result = await db.run(
      'DELETE FROM user_wanted_skills WHERE user_id = ? AND skill_id = ?',
      [userId, skillId]
    );

    if (result.changes === 0) {
      return res.status(404).json({ error: 'Wanted skill not found' });
    }

    res.json({ message: 'Wanted skill removed successfully' });
  } catch (error) {
    console.error('Remove wanted skill error:', error);
    res.status(500).json({ error: 'Server error while removing wanted skill' });
  }
});

module.exports = router;