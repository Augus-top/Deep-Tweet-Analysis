const { Pool, Client } = require('pg');

const config = require('../config.json');

const pool = new Pool({
  user: config.db_user,
  host: config.host,
  database: config.db_name,
  password: config.db_pass,
  port: config.db_port
});

pool.on('error', (err) => {
  console.error('Unexpected error on idle database client class: ', err);
  process.exit(-1);
});

exports.executeQuery = async (query, values) => {
  const client = await pool.connect();
  try {
    const res = await client.query(query, values);
    return res.rows;
  } catch (err) {
    console.log(err);
    return 'error';
  } finally {
    client.release();
  }
};

exports.saveTweets = async (tweets, tweetQuery, sentimentLabel) => {
  const dbQuery = 'INSERT INTO coleta_sentimentos (id, tweet_text, tweet_date, sentiment, query_used) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (id) DO NOTHING';
  tweets.forEach((tweet) => {
    const dbValues = [tweet.id_str, tweet.full_text, tweet.created_at, sentimentLabel, tweetQuery];
    this.executeQuery(dbQuery, dbValues);
  });
};

exports.findSinceId = async (tweetQuery) => {
  const dbQuery = 'SELECT id FROM coleta_sentimentos WHERE query_used = $1 ORDER BY id limit 1';
  const dbValues = [tweetQuery];
  const res = await this.executeQuery(dbQuery, dbValues);
  return (res.length > 0) ? res[0].id : undefined;
};

