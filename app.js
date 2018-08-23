const express = require('express');
const path = require('path');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');

const index = require('./routes/index');
const tweetAPI = require('./routes/tweetRoute');
const tweetWatcher = require('./controllers/tweetController');

const app = express();
const port = process.env.PORT || '3030';

app.set('views', path.join(__dirname, 'views'));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', index);
app.use('/api/tweet', tweetAPI);

app.use((req, res) => {
  res.status(404).send({ url: `${req.originalUrl} not found` });
});

const server = app.listen(port, function () {});
console.log(`Connected on port ${port}`);

// tweetWatcher.realizeSearchForCandidates();
// tweetWatcher.realizeSearchForHashtags();
// tweetWatcher.realizeSearchForPoliticalTerms();
// tweetWatcher.realizeSearchForExtraTerms();
// tweetWatcher.realizeSearchForExtraTerms2();
// tweetWatcher.collectEmoticonWithNoScope(':)', 'Positivo');
// tweetWatcher.collectEmoticonWithNoScope(':(', 'Negativo');
// tweetWatcher.checkRateLimit();

module.exports = app;
