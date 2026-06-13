// Authentication middleware
// BUG: Checks req.headers.authorization but login sets x-auth-token
function authMiddleware(req, res, next) {
  const token = req.headers.authorization;

  if (!token) {
    return res.status(401).json({ error: 'No token provided' });
  }

  try {
    const decoded = Buffer.from(token, 'base64').toString('utf8');
    const [userId, timestamp] = decoded.split(':');

    if (!userId || !timestamp) {
      return res.status(401).json({ error: 'Invalid token format' });
    }

    // Token expires after 1 hour
    if (Date.now() - parseInt(timestamp) > 3600000) {
      return res.status(401).json({ error: 'Token expired' });
    }

    req.userId = parseInt(userId);
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid token' });
  }
}

module.exports = authMiddleware;
