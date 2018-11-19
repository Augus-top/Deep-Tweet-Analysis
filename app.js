const express = require('express');
const path = require('path');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');

const index = require('./routes/index');

const app = express();
const port = process.env.PORT || '3000';

app.set('views', path.join(__dirname, 'views'));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', index);

app.use((req, res) => {
  res.status(404).send({ url: `${req.originalUrl} not found` });
});

const server = app.listen(port, function () {});
console.log(`Connected on port ${port}`);

module.exports = app;
