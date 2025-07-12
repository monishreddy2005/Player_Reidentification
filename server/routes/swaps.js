const express = require('express');
const { body, validationResult } = require('express-validator');
const db = require('../config/database');
const { authenticateToken, requireAdmin } = require('../middleware/auth');

const router = express.Router();

// Get user's swap requests (sent and received)
router.get('/my', authenticateToken, async (req, res) => {
  try {
    const { status, type = 'all', page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    let query = `
      SELECT sr.*, 
             requester.name as requester_name, requester.username as requester_username, requester.profile_photo as requester_photo,
             provider.name as provider_name, provider.username as provider_username, provider.profile_photo as provider_photo,
             rs.name as requested_skill_name, rs.category as requested_skill_category,
             os.name as offered_skill_name, os.category as offered_skill_category
      FROM swap_requests sr
      JOIN users requester ON sr.requester_id = requester.id
      JOIN users provider ON sr.provider_id = provider.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      WHERE (sr.requester_id = ? OR sr.provider_id = ?)
    `;

    const params = [req.user.id, req.user.id];

    if (status) {
      query += ' AND sr.status = ?';
      params.push(status);
    }

    if (type === 'sent') {
      query += ' AND sr.requester_id = ?';
      params.push(req.user.id);
    } else if (type === 'received') {
      query += ' AND sr.provider_id = ?';
      params.push(req.user.id);
    }

    query += ' ORDER BY sr.created_at DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);

    const swapRequests = await db.query(query, params);

    res.json({ swapRequests, page: parseInt(page), limit: parseInt(limit) });
  } catch (error) {
    console.error('Get my swaps error:', error);
    res.status(500).json({ error: 'Server error while fetching swap requests' });
  }
});

// Get swap request by ID
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const swapId = req.params.id;

    const swapRequest = await db.get(`
      SELECT sr.*, 
             requester.name as requester_name, requester.username as requester_username, 
             requester.profile_photo as requester_photo, requester.email as requester_email,
             provider.name as provider_name, provider.username as provider_username, 
             provider.profile_photo as provider_photo, provider.email as provider_email,
             rs.name as requested_skill_name, rs.category as requested_skill_category,
             os.name as offered_skill_name, os.category as offered_skill_category
      FROM swap_requests sr
      JOIN users requester ON sr.requester_id = requester.id
      JOIN users provider ON sr.provider_id = provider.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
      WHERE sr.id = ?
    `, [swapId]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Check if user has permission to view this swap
    if (swapRequest.requester_id !== req.user.id && 
        swapRequest.provider_id !== req.user.id && 
        !req.user.is_admin) {
      return res.status(403).json({ error: 'Access denied' });
    }

    res.json(swapRequest);
  } catch (error) {
    console.error('Get swap request error:', error);
    res.status(500).json({ error: 'Server error while fetching swap request' });
  }
});

// Create swap request
router.post('/', [
  authenticateToken,
  body('provider_id').isInt(),
  body('requested_skill_id').isInt(),
  body('offered_skill_id').isInt(),
  body('message').optional().isLength({ max: 500 }).trim().escape()
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { provider_id, requested_skill_id, offered_skill_id, message } = req.body;

    // Validate that user is not requesting from themselves
    if (provider_id == req.user.id) {
      return res.status(400).json({ error: 'Cannot request skill swap from yourself' });
    }

    // Check if provider exists and is not banned
    const provider = await db.get('SELECT id, is_banned FROM users WHERE id = ?', [provider_id]);
    if (!provider) {
      return res.status(404).json({ error: 'Provider not found' });
    }
    if (provider.is_banned) {
      return res.status(400).json({ error: 'Provider is banned' });
    }

    // Check if skills exist
    const [requestedSkill, offeredSkill] = await Promise.all([
      db.get('SELECT id FROM skills WHERE id = ?', [requested_skill_id]),
      db.get('SELECT id FROM skills WHERE id = ?', [offered_skill_id])
    ]);

    if (!requestedSkill || !offeredSkill) {
      return res.status(404).json({ error: 'One or both skills not found' });
    }

    // Check if provider actually offers the requested skill
    const providerSkill = await db.get(
      'SELECT id FROM user_offered_skills WHERE user_id = ? AND skill_id = ?',
      [provider_id, requested_skill_id]
    );

    if (!providerSkill) {
      return res.status(400).json({ error: 'Provider does not offer the requested skill' });
    }

    // Check if requester actually offers the offered skill
    const requesterSkill = await db.get(
      'SELECT id FROM user_offered_skills WHERE user_id = ? AND skill_id = ?',
      [req.user.id, offered_skill_id]
    );

    if (!requesterSkill) {
      return res.status(400).json({ error: 'You do not offer the specified skill' });
    }

    // Check if there's already a pending request between these users for these skills
    const existingRequest = await db.get(`
      SELECT id FROM swap_requests 
      WHERE requester_id = ? AND provider_id = ? AND requested_skill_id = ? AND offered_skill_id = ?
      AND status = 'pending'
    `, [req.user.id, provider_id, requested_skill_id, offered_skill_id]);

    if (existingRequest) {
      return res.status(409).json({ error: 'A pending request already exists for these skills' });
    }

    // Create swap request
    const result = await db.run(
      'INSERT INTO swap_requests (requester_id, provider_id, requested_skill_id, offered_skill_id, message) VALUES (?, ?, ?, ?, ?)',
      [req.user.id, provider_id, requested_skill_id, offered_skill_id, message]
    );

    res.status(201).json({ 
      message: 'Swap request created successfully',
      id: result.lastID
    });
  } catch (error) {
    console.error('Create swap request error:', error);
    res.status(500).json({ error: 'Server error while creating swap request' });
  }
});

// Accept swap request
router.post('/:id/accept', authenticateToken, async (req, res) => {
  try {
    const swapId = req.params.id;

    const swapRequest = await db.get('SELECT * FROM swap_requests WHERE id = ?', [swapId]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Only the provider can accept the request
    if (swapRequest.provider_id !== req.user.id) {
      return res.status(403).json({ error: 'Only the provider can accept this request' });
    }

    // Check if request is still pending
    if (swapRequest.status !== 'pending') {
      return res.status(400).json({ error: 'Request is no longer pending' });
    }

    // Update request status
    await db.run(
      'UPDATE swap_requests SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
      ['accepted', swapId]
    );

    res.json({ message: 'Swap request accepted successfully' });
  } catch (error) {
    console.error('Accept swap request error:', error);
    res.status(500).json({ error: 'Server error while accepting swap request' });
  }
});

// Reject swap request
router.post('/:id/reject', authenticateToken, async (req, res) => {
  try {
    const swapId = req.params.id;

    const swapRequest = await db.get('SELECT * FROM swap_requests WHERE id = ?', [swapId]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Only the provider can reject the request
    if (swapRequest.provider_id !== req.user.id) {
      return res.status(403).json({ error: 'Only the provider can reject this request' });
    }

    // Check if request is still pending
    if (swapRequest.status !== 'pending') {
      return res.status(400).json({ error: 'Request is no longer pending' });
    }

    // Update request status
    await db.run(
      'UPDATE swap_requests SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
      ['rejected', swapId]
    );

    res.json({ message: 'Swap request rejected successfully' });
  } catch (error) {
    console.error('Reject swap request error:', error);
    res.status(500).json({ error: 'Server error while rejecting swap request' });
  }
});

// Cancel swap request (by requester)
router.post('/:id/cancel', authenticateToken, async (req, res) => {
  try {
    const swapId = req.params.id;

    const swapRequest = await db.get('SELECT * FROM swap_requests WHERE id = ?', [swapId]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Only the requester can cancel the request
    if (swapRequest.requester_id !== req.user.id) {
      return res.status(403).json({ error: 'Only the requester can cancel this request' });
    }

    // Check if request is still pending
    if (swapRequest.status !== 'pending') {
      return res.status(400).json({ error: 'Request is no longer pending' });
    }

    // Update request status
    await db.run(
      'UPDATE swap_requests SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
      ['cancelled', swapId]
    );

    res.json({ message: 'Swap request cancelled successfully' });
  } catch (error) {
    console.error('Cancel swap request error:', error);
    res.status(500).json({ error: 'Server error while cancelling swap request' });
  }
});

// Delete swap request (only if not accepted)
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const swapId = req.params.id;

    const swapRequest = await db.get('SELECT * FROM swap_requests WHERE id = ?', [swapId]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Only the requester or admin can delete
    if (swapRequest.requester_id !== req.user.id && !req.user.is_admin) {
      return res.status(403).json({ error: 'Access denied' });
    }

    // Cannot delete accepted requests
    if (swapRequest.status === 'accepted') {
      return res.status(400).json({ error: 'Cannot delete accepted swap requests' });
    }

    // Delete the request
    await db.run('DELETE FROM swap_requests WHERE id = ?', [swapId]);

    res.json({ message: 'Swap request deleted successfully' });
  } catch (error) {
    console.error('Delete swap request error:', error);
    res.status(500).json({ error: 'Server error while deleting swap request' });
  }
});

// Mark swap as completed (by either party)
router.post('/:id/complete', authenticateToken, async (req, res) => {
  try {
    const swapId = req.params.id;

    const swapRequest = await db.get('SELECT * FROM swap_requests WHERE id = ?', [swapId]);

    if (!swapRequest) {
      return res.status(404).json({ error: 'Swap request not found' });
    }

    // Only involved parties can mark as completed
    if (swapRequest.requester_id !== req.user.id && swapRequest.provider_id !== req.user.id) {
      return res.status(403).json({ error: 'Only involved parties can mark swap as completed' });
    }

    // Check if request is accepted
    if (swapRequest.status !== 'accepted') {
      return res.status(400).json({ error: 'Swap must be accepted before it can be completed' });
    }

    // Update request status
    await db.run(
      'UPDATE swap_requests SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
      ['completed', swapId]
    );

    res.json({ message: 'Swap marked as completed successfully' });
  } catch (error) {
    console.error('Complete swap request error:', error);
    res.status(500).json({ error: 'Server error while completing swap request' });
  }
});

// Get all swap requests (admin only)
router.get('/admin/all', authenticateToken, requireAdmin, async (req, res) => {
  try {
    const { status, page = 1, limit = 20 } = req.query;
    const offset = (page - 1) * limit;

    let query = `
      SELECT sr.*, 
             requester.name as requester_name, requester.username as requester_username,
             provider.name as provider_name, provider.username as provider_username,
             rs.name as requested_skill_name, os.name as offered_skill_name
      FROM swap_requests sr
      JOIN users requester ON sr.requester_id = requester.id
      JOIN users provider ON sr.provider_id = provider.id
      JOIN skills rs ON sr.requested_skill_id = rs.id
      JOIN skills os ON sr.offered_skill_id = os.id
    `;

    const params = [];

    if (status) {
      query += ' WHERE sr.status = ?';
      params.push(status);
    }

    query += ' ORDER BY sr.created_at DESC LIMIT ? OFFSET ?';
    params.push(parseInt(limit), offset);

    const swapRequests = await db.query(query, params);

    // Get counts by status
    const statusCounts = await db.query(`
      SELECT status, COUNT(*) as count
      FROM swap_requests
      GROUP BY status
    `);

    res.json({ 
      swapRequests, 
      statusCounts: statusCounts.reduce((acc, item) => {
        acc[item.status] = item.count;
        return acc;
      }, {}),
      page: parseInt(page), 
      limit: parseInt(limit) 
    });
  } catch (error) {
    console.error('Get all swaps error:', error);
    res.status(500).json({ error: 'Server error while fetching all swap requests' });
  }
});

module.exports = router;