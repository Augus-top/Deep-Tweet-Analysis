const express = require('express');
const path = require('path');

const router = express.Router();

/* GET home page. */
router.get('/', (req, res) => {
  res.send('<h1>Deep Learning Analysis</h1>');
});

router.get('/model2/:name', (req, res) => {
  res.sendFile(path.join(__dirname, '../models/' + req.params.name));
});

router.use((error, req, res, next) => {
  res.status(404).json({
    type: 'FileNotFound',
    message: req.originalUrl + ' was not found'
  });
});

module.exports = router;
