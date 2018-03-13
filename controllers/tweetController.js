const Twit = require('twit');
const config = require('../config.json');

const QUERY_LIMIT = 100;
const twit = new Twit({
  consumer_key: config.key,
  consumer_secret: config.secret,
  access_token: config.token,
  access_token_secret: config.token_secret,
  timeout_ms: 60 * 1000
});

exports.searchQuery = async (query, numberOfTweets = QUERY_LIMIT) => {
  try {
    const twitterResponse = await twit.get('search/tweets', { q: query, count: 1 });
    console.log(twitterResponse.data.statuses[0].text);
  } catch (err) {
    console.log(err);
  }
};
