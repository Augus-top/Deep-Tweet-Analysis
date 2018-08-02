const express = require('express');
const controller = require('../controllers/tweetController');

const router = express.Router();

router.get('/', async (req, res, next) => {
  controller.searchQuery('univali');
  res.send('ok');
});

router.get('/buscar/termos', async (req, res, next) => {
  controller.realizeSearchForPoliticalTerms();
  res.send('ok');
});

router.get('/buscar/candidatos', async (req, res, next) => {
  controller.realizeSearchForCandidates();
  res.send('ok');
});
router.get('/buscar/hashtags', async (req, res, next) => {
  controller.realizeSearchForHashtags();
  res.send('ok');
});

module.exports = router;
