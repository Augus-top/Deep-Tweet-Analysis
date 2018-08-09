const fs = require('fs');
const csvWriterStream = require('csv-write-stream');

exports.saveTweets = async (tweets, query, sentimentLabel) => {
  const fileName = this.determineFileName(query);
  const csvWriter = csvWriterStream({ sendHeaders: false });
  csvWriter.pipe(fs.createWriteStream(`./Coleta/${fileName}`, { flags: 'a' }));
  tweets.forEach((tweet) => {
    csvWriter.write({
      id_str: tweet.id_str,
      text: tweet.full_text,
      date: tweet.created_at,
      sentiment: sentimentLabel
    });
  });
  csvWriter.end();
};

exports.findSinceId = () => {
  return undefined;
};

exports.saveStreamInCSV = async (stream, csvFileName, sentimentLabel) => {
  const streamText = (stream.extended_tweet === undefined) ? stream.text : stream.extended_tweet.full_text;
  const tweet = {
    id_str: stream.id_str,
    full_text: streamText,
    date: stream.created_at
  };
  this.saveTweets([tweet], csvFileName, sentimentLabel);
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
