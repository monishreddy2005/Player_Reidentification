const express = require('express');
const { body, validationResult } = require('express-validator');
const db = require('../config/database');
const { authenticateToken, requireAdmin } = require('../middleware/auth');

const router = express.Router();

// Get all skills
router.get('/', async (req, res) => {
  try {
    const { search, category, page = 1, limit = 50 } = req.query;
    const offset = (page - 1) * limit;

    let query = 'SELECT * FROM skills WHERE is_approved = 1';
    const params = [];

    if (search) {
      query += ' AND name LIKE ?';
      params.push(`%${search}%`);
    }

    if (category) {
      query += ' AND category = ?';
      params.push(category);
    }

    query += ' ORDER BY name ASC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);

    const skills = await db.query(query, params);

    // Get skill categories
    const categories = await db.query('SELECT DISTINCT category FROM skills WHERE is_approved = 1 ORDER BY category');

    res.json({ 
      skills, 
      categories: categories.map(c => c.category).filter(Boolean),
      page: parseInt(page), 
      limit: parseInt(limit) 
    });
  } catch (error) {
    console.error('Get skills error:', error);
    res.status(500).json({ error: 'Server error while fetching skills' });
  }
});

// Get skill by ID
router.get('/:id', async (req, res) => {
  try {
    const skillId = req.params.id;

    const skill = await db.get('SELECT * FROM skills WHERE id = ? AND is_approved = 1', [skillId]);

    if (!skill) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    // Get users who offer this skill
    const offeredBy = await db.query(`
      SELECT u.id, u.name, u.username, u.location, u.profile_photo, uos.proficiency_level
      FROM user_offered_skills uos
      JOIN users u ON uos.user_id = u.id
      WHERE uos.skill_id = ? AND u.is_public = 1 AND u.is_banned = 0
      ORDER BY u.name
    `, [skillId]);

    // Get users who want this skill
    const wantedBy = await db.query(`
      SELECT u.id, u.name, u.username, u.location, u.profile_photo, uws.urgency
      FROM user_wanted_skills uws
      JOIN users u ON uws.user_id = u.id
      WHERE uws.skill_id = ? AND u.is_public = 1 AND u.is_banned = 0
      ORDER BY u.name
    `, [skillId]);

    skill.offeredBy = offeredBy;
    skill.wantedBy = wantedBy;

    res.json(skill);
  } catch (error) {
    console.error('Get skill error:', error);
    res.status(500).json({ error: 'Server error while fetching skill' });
  }
});

// Create new skill (authenticated users)
router.post('/', [
  authenticateToken,
  body('name').isLength({ min: 2, max: 100 }).trim().escape(),
  body('category').optional().isLength({ max: 50 }).trim().escape(),
  body('description').optional().isLength({ max: 500 }).trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { name, category, description } = req.body;

    // Check if skill already exists
    const existingSkill = await db.get('SELECT id FROM skills WHERE name = ?', [name]);
    if (existingSkill) {
      return res.status(409).json({ error: 'Skill already exists' });
    }

    // Create skill (needs admin approval for new skills)
    const result = await db.run(
      'INSERT INTO skills (name, category, description, is_approved) VALUES (?, ?, ?, ?)',
      [name, category, description, req.user.is_admin ? 1 : 0]
    );

    res.status(201).json({ 
      message: req.user.is_admin ? 'Skill created successfully' : 'Skill submitted for approval',
      id: result.lastID
    });
  } catch (error) {
    console.error('Create skill error:', error);
    res.status(500).json({ error: 'Server error while creating skill' });
  }
});

// Update skill (admin only)
router.put('/:id', [
  authenticateToken,
  requireAdmin,
  body('name').optional().isLength({ min: 2, max: 100 }).trim().escape(),
  body('category').optional().isLength({ max: 50 }).trim().escape(),
  body('description').optional().isLength({ max: 500 }).trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const skillId = req.params.id;
    const { name, category, description } = req.body;

    const updates = [];
    const params = [];

    if (name !== undefined) {
      updates.push('name = ?');
      params.push(name);
    }
    if (category !== undefined) {
      updates.push('category = ?');
      params.push(category);
    }
    if (description !== undefined) {
      updates.push('description = ?');
      params.push(description);
    }

    if (updates.length === 0) {
      return res.status(400).json({ error: 'No valid fields to update' });
    }

    params.push(skillId);

    const result = await db.run(
      `UPDATE skills SET ${updates.join(', ')} WHERE id = ?`,
      params
    );

    if (result.changes === 0) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    res.json({ message: 'Skill updated successfully' });
  } catch (error) {
    console.error('Update skill error:', error);
    res.status(500).json({ error: 'Server error while updating skill' });
  }
});

// Delete skill (admin only)
router.delete('/:id', authenticateToken, requireAdmin, async (req, res) => {
  try {
    const skillId = req.params.id;

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

    const result = await db.run('DELETE FROM skills WHERE id = ?', [skillId]);

    if (result.changes === 0) {
      return res.status(404).json({ error: 'Skill not found' });
    }

    res.json({ message: 'Skill deleted successfully' });
  } catch (error) {
    console.error('Delete skill error:', error);
    res.status(500).json({ error: 'Server error while deleting skill' });
  }
});

// Search skills with user context
router.get('/search/users', authenticateToken, async (req, res) => {
  try {
    const { skill, location, page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    if (!skill) {
      return res.status(400).json({ error: 'Skill parameter required' });
    }

    let query = `
      SELECT DISTINCT u.id, u.name, u.username, u.location, u.profile_photo, u.bio, u.availability,
             s.name as skill_name, s.category, uos.proficiency_level, uos.description as skill_description
      FROM users u
      JOIN user_offered_skills uos ON u.id = uos.user_id
      JOIN skills s ON uos.skill_id = s.id
      WHERE u.is_banned = 0 AND u.is_public = 1 AND u.id != ? AND s.name LIKE ?
    `;

    const params = [req.user.id, `%${skill}%`];

    if (location) {
      query += ' AND u.location LIKE ?';
      params.push(`%${location}%`);
    }

    query += ' ORDER BY u.name LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);

    const users = await db.query(query, params);

    // Get average ratings for each user
    for (let user of users) {
      const rating = await db.get(`
        SELECT AVG(rating) as average_rating, COUNT(*) as total_ratings
        FROM ratings
        WHERE rated_id = ?
      `, [user.id]);

      user.averageRating = rating.average_rating;
      user.totalRatings = rating.total_ratings;
    }

    res.json({ users, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Search skills error:', error);
    res.status(500).json({ error: 'Server error while searching skills' });
  }
});

module.exports = router;