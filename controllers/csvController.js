const fs = require('fs');
const csvWriterStream = require('csv-write-stream');

exports.saveTweetsInCSV = async (tweets, csvFileName, sentimentLabel) => {
  const csvWriter = csvWriterStream({ sendHeaders: false });
  csvWriter.pipe(fs.createWriteStream(`./Coleta/${csvFileName}`, { flags: 'a' }));
  const tweetsToSave = tweets.filter(tweet => !tweet.full_text.startsWith('RT @'));
  tweetsToSave.forEach((tweet) => {
    tweet.text = tweet.full_text.replace(/(\r\n\t|\n|\r\t)/gm, ' '); // Newlines
    tweet.text = tweet.full_text.replace(/\s\s+/g, ' '); // Multiple spaces
    csvWriter.write({
      id_str: tweet.id_str,
      text: tweet.full_text,
      date: tweet.created_at,
      sentiment: sentimentLabel
    });
  });
  csvWriter.end();
};

exports.saveStreamInCSV = async (stream, csvFileName, sentimentLabel) => {
  const streamText = (stream.extended_tweet === undefined) ? stream.text : stream.extended_tweet.full_text;
  const tweet = {
    id_str: stream.id_str,
    full_text: streamText,
    date: stream.created_at
  };
  this.saveTweetsInCSV([tweet], csvFileName, sentimentLabel);
};

exports.determineFileName = (query) => {
  let fileName = '';
  if (query.startsWith('#')) {
    fileName = 'Hash';
    query = query.slice(1);
  }
  const firstSpace = query.indexOf(' ');
  fileName += (firstSpace === -1)
    ? query.slice(0, 1).toUpperCase() + query.slice(1)
    : query.slice(0, 1).toUpperCase() + query.slice(1, firstSpace);
  return `${fileName}.csv`;
};
