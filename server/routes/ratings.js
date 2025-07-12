const express = require('express');
const { body, validationResult } = require('express-validator');
const db = require('../config/database');
const { authenticateToken, requireAdmin } = require('../middleware/auth');

const router = express.Router();

// Create rating/feedback
router.post('/', [
  authenticateToken,
  body('swap_request_id').isInt(),
  body('rated_id').isInt(),
  body('rating').isInt({ min: 1, max: 5 }),
  body('feedback').optional().isLength({ max: 500 }).trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { swap_request_id, rated_id, rating, feedback } = req.body;

    // Get swap request details
    const swapRequest = await db.get('SELECT * FROM swap_requests WHERE id = ?', [swap_request_id]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Check if swap is completed
    if (swapRequest.status !== 'completed') {
      return res.status(400).json({ error: 'Can only rate completed swaps' });
    }

    // Check if user is part of the swap
    if (swapRequest.requester_id !== req.user.id && swapRequest.provider_id !== req.user.id) {
      return res.status(403).json({ error: 'You can only rate swaps you participated in' });
    }

    // Check if user is trying to rate the correct person
    const otherUserId = swapRequest.requester_id === req.user.id ? swapRequest.provider_id : swapRequest.requester_id;
    if (rated_id !== otherUserId) {
      return res.status(400).json({ error: 'You can only rate the other party in the swap' });
    }

    // Check if user has already rated this swap
    const existingRating = await db.get(
      'SELECT id FROM ratings WHERE swap_request_id = ? AND rater_id = ?',
      [swap_request_id, req.user.id]
    );

    if (existingRating) {
      return res.status(409).json({ error: 'You have already rated this swap' });
    }

    // Create rating
    const result = await db.run(
      'INSERT INTO ratings (swap_request_id, rater_id, rated_id, rating, feedback) VALUES (?, ?, ?, ?, ?)',
      [swap_request_id, req.user.id, rated_id, rating, feedback]
    );

    res.status(201).json({ 
      message: 'Rating submitted successfully',
      id: result.lastID
    });
  } catch (error) {
    console.error('Create rating error:', error);
    res.status(500).json({ error: 'Server error while creating rating' });
  }
});

// Get user's ratings (received)
router.get('/user/:userId', async (req, res) => {
  try {
    const userId = req.params.userId;
    const { page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    // Get ratings for the user
    const ratings = await db.query(`
      SELECT r.*, 
             rater.name as rater_name, rater.username as rater_username, rater.profile_photo as rater_photo,
             sr.requested_skill_id, sr.offered_skill_id,
             rs.name as requested_skill_name, os.name as offered_skill_name
      FROM ratings r
      JOIN users rater ON r.rater_id = rater.id
      JOIN swap_requests sr ON r.swap_request_id = sr.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      WHERE r.rated_id = ?
      ORDER BY r.created_at DESC
      LIMIT ? OFFSET ?
    `, [userId, parseInt(limit), offset]);

    // Get rating summary
    const summary = await db.get(`
      SELECT 
        AVG(rating) as average_rating,
        COUNT(*) as total_ratings,
        COUNT(CASE WHEN rating = 5 THEN 1 END) as five_star,
        COUNT(CASE WHEN rating = 4 THEN 1 END) as four_star,
        COUNT(CASE WHEN rating = 3 THEN 1 END) as three_star,
        COUNT(CASE WHEN rating = 2 THEN 1 END) as two_star,
        COUNT(CASE WHEN rating = 1 THEN 1 END) as one_star
      FROM ratings
      WHERE rated_id = ?
    `, [userId]);

    res.json({ 
      ratings, 
      summary,
      page: parseInt(page), 
      limit: parseInt(limit) 
    });
  } catch (error) {
    console.error('Get user ratings error:', error);
    res.status(500).json({ error: 'Server error while fetching user ratings' });
  }
});

// Get rating by ID
router.get('/:id', async (req, res) => {
  try {
    const ratingId = req.params.id;

    const rating = await db.get(`
      SELECT r.*, 
             rater.name as rater_name, rater.username as rater_username, rater.profile_photo as rater_photo,
             rated.name as rated_name, rated.username as rated_username, rated.profile_photo as rated_photo,
             sr.requested_skill_id, sr.offered_skill_id,
             rs.name as requested_skill_name, os.name as offered_skill_name
      FROM ratings r
      JOIN users rater ON r.rater_id = rater.id
      JOIN users rated ON r.rated_id = rated.id
      JOIN swap_requests sr ON r.swap_request_id = sr.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      WHERE r.id = ?
    `, [ratingId]);

    if (!rating) {
      return res.status(404).json({ error: 'Rating not found' });
    }

    res.json(rating);
  } catch (error) {
    console.error('Get rating error:', error);
    res.status(500).json({ error: 'Server error while fetching rating' });
  }
});

// Update rating (only by rater within 24 hours)
router.put('/:id', [
  authenticateToken,
  body('rating').optional().isInt({ min: 1, max: 5 }),
  body('feedback').optional().isLength({ max: 500 }).trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const ratingId = req.params.id;
    const { rating, feedback } = req.body;

    // Get existing rating
    const existingRating = await db.get('SELECT * FROM ratings WHERE id = ?', [ratingId]);

    if (!existingRating) {
      return res.status(404).json({ error: 'Rating not found' });
    }

    // Check if user is the rater
    if (existingRating.rater_id !== req.user.id) {
      return res.status(403).json({ error: 'You can only update your own ratings' });
    }

    // Check if rating is within 24 hours
    const ratingDate = new Date(existingRating.created_at);
    const now = new Date();
    const timeDiff = now - ratingDate;
    const hours = timeDiff / (1000 * 60 * 60);

    if (hours > 24) {
      return res.status(400).json({ error: 'Ratings can only be updated within 24 hours' });
    }

    const updates = [];
    const params = [];

    if (rating !== undefined) {
      updates.push('rating = ?');
      params.push(rating);
    }
    if (feedback !== undefined) {
      updates.push('feedback = ?');
      params.push(feedback);
    }

    if (updates.length === 0) {
      return res.status(400).json({ error: 'No valid fields to update' });
    }

    params.push(ratingId);

    await db.run(
      `UPDATE ratings SET ${updates.join(', ')} WHERE id = ?`,
      params
    );

    res.json({ message: 'Rating updated successfully' });
  } catch (error) {
    console.error('Update rating error:', error);
    res.status(500).json({ error: 'Server error while updating rating' });
  }
});

// Delete rating (only by rater within 24 hours or admin)
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const ratingId = req.params.id;

    // Get existing rating
    const existingRating = await db.get('SELECT * FROM ratings WHERE id = ?', [ratingId]);

    if (!existingRating) {
      return res.status(404).json({ error: 'Rating not found' });
    }

    // Check permissions
    if (existingRating.rater_id !== req.user.id && !req.user.is_admin) {
      return res.status(403).json({ error: 'You can only delete your own ratings' });
    }

    // Check if rating is within 24 hours (unless admin)
    if (!req.user.is_admin) {
      const ratingDate = new Date(existingRating.created_at);
      const now = new Date();
      const timeDiff = now - ratingDate;
      const hours = timeDiff / (1000 * 60 * 60);

      if (hours > 24) {
        return res.status(400).json({ error: 'Ratings can only be deleted within 24 hours' });
      }
    }

    // Delete rating
    await db.run('DELETE FROM ratings WHERE id = ?', [ratingId]);

    res.json({ message: 'Rating deleted successfully' });
  } catch (error) {
    console.error('Delete rating error:', error);
    res.status(500).json({ error: 'Server error while deleting rating' });
  }
});

// Get my ratings (given by current user)
router.get('/my/given', authenticateToken, async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    const ratings = await db.query(`
      SELECT r.*, 
             rated.name as rated_name, rated.username as rated_username, rated.profile_photo as rated_photo,
             sr.requested_skill_id, sr.offered_skill_id,
             rs.name as requested_skill_name, os.name as offered_skill_name
      FROM ratings r
      JOIN users rated ON r.rated_id = rated.id
      JOIN swap_requests sr ON r.swap_request_id = sr.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      WHERE r.rater_id = ?
      ORDER BY r.created_at DESC
      LIMIT ? OFFSET ?
    `, [req.user.id, parseInt(limit), offset]);

    res.json({ ratings, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get my given ratings error:', error);
    res.status(500).json({ error: 'Server error while fetching given ratings' });
  }
});

// Get my ratings (received by current user)
router.get('/my/received', authenticateToken, async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    const ratings = await db.query(`
      SELECT r.*, 
             rater.name as rater_name, rater.username as rater_username, rater.profile_photo as rater_photo,
             sr.requested_skill_id, sr.offered_skill_id,
             rs.name as requested_skill_name, os.name as offered_skill_name
      FROM ratings r
      JOIN users rater ON r.rater_id = rater.id
      JOIN swap_requests sr ON r.swap_request_id = sr.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      WHERE r.rated_id = ?
      ORDER BY r.created_at DESC
      LIMIT ? OFFSET ?
    `, [req.user.id, parseInt(limit), offset]);

    // Get rating summary
    const summary = await db.get(`
      SELECT 
        AVG(rating) as average_rating,
        COUNT(*) as total_ratings,
        COUNT(CASE WHEN rating = 5 THEN 1 END) as five_star,
        COUNT(CASE WHEN rating = 4 THEN 1 END) as four_star,
        COUNT(CASE WHEN rating = 3 THEN 1 END) as three_star,
        COUNT(CASE WHEN rating = 2 THEN 1 END) as two_star,
        COUNT(CASE WHEN rating = 1 THEN 1 END) as one_star
      FROM ratings
      WHERE rated_id = ?
    `, [req.user.id]);

    res.json({ 
      ratings, 
      summary,
      page: parseInt(page), 
      limit: parseInt(limit) 
    });
  } catch (error) {
    console.error('Get my received ratings error:', error);
    res.status(500).json({ error: 'Server error while fetching received ratings' });
  }
});

// Get all ratings (admin only)
router.get('/admin/all', authenticateToken, requireAdmin, async (req, res) => {
  try {
    const { page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;

    const ratings = await db.query(`
      SELECT r.*, 
             rater.name as rater_name, rater.username as rater_username,
             rated.name as rated_name, rated.username as rated_username,
             sr.requested_skill_id, sr.offered_skill_id,
             rs.name as requested_skill_name, os.name as offered_skill_name
      FROM ratings r
      JOIN users rater ON r.rater_id = rater.id
      JOIN users rated ON r.rated_id = rated.id
      JOIN swap_requests sr ON r.swap_request_id = sr.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      ORDER BY r.created_at DESC
      LIMIT ? OFFSET ?
    `, [parseInt(limit), offset]);

    // Get rating statistics
    const stats = await db.get(`
      SELECT 
        COUNT(*) as total_ratings,
        AVG(rating) as average_rating,
        COUNT(CASE WHEN rating = 5 THEN 1 END) as five_star,
        COUNT(CASE WHEN rating = 4 THEN 1 END) as four_star,
        COUNT(CASE WHEN rating = 3 THEN 1 END) as three_star,
        COUNT(CASE WHEN rating = 2 THEN 1 END) as two_star,
        COUNT(CASE WHEN rating = 1 THEN 1 END) as one_star
      FROM ratings
    `);

    res.json({ 
      ratings, 
      stats,
      page: parseInt(page), 
      limit: parseInt(limit) 
    });
  } catch (error) {
    console.error('Get all ratings error:', error);
    res.status(500).json({ error: 'Server error while fetching all ratings' });
  }
});

module.exports = router;