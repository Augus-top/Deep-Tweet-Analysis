const express = require('express');
const controller = require('../controllers/tweetController');

const router = express.Router();

router.get('/', async (req, res, next) => {
  controller.searchQuery('banana');
  res.send('ok');
});

module.exports = router;
