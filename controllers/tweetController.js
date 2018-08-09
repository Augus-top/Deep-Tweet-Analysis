const Twit = require('twit');

const csvController = require('./csvController.js');
const dbController = require('./dbController');
const config = require('../config.json');

const CURRENT_WRITTER = 'db'; // or csv
const QUERY_LIMIT = 100;
const twit = new Twit({
  consumer_key: config.key,
  consumer_secret: config.secret,
  access_token: config.token,
  access_token_secret: config.token_secret,
  timeout_ms: 60 * 1000
});

const clearTweets = (tweets) => {
  const clearedTweets = tweets.filter(tweet => !tweet.full_text.startsWith('RT @')).map((tweet) => {
    tweet.full_text = tweet.full_text.replace(/(\r\n\t|\n|\r\t)/gm, ' '); // Newlines
    tweet.full_text = tweet.full_text.replace(/\s\s+/g, ' '); // Multiple spaces
    return tweet;
  });
  return clearedTweets;
};

const defineWritter = (writterName) => {
  const writter = (writterName === 'db') ? dbController : csvController;
  return writter;
};

exports.generateSentimentQueries = (terms) => {
  const tweetWritter = defineWritter(CURRENT_WRITTER);
  terms.forEach((term) => {
    this.searchQuery(term.name + ' :)', 'Positivo', tweetWritter);
    this.searchQuery(term.name + ' :(', 'Negativo', tweetWritter);
  });
};

exports.realizeSearchForCandidates = async () => {
  const terms = [
    { name: 'alckmin' },
    { name: 'bolsonaro' },
    { name: 'hassad' },
    { name: 'lula' },
    { name: 'aldo rebelo' },
    { name: 'alvaro dias' },
    { name: 'ciro' },
    { name: 'collor' },
    { name: 'flavio rocha' },
    { name: 'guilherme boulos' },
    { name: 'henrique meirelles' },
    { name: 'joao amoedo' },
    { name: 'levy fidelix' },
    { name: "manuela d'avila" },
    { name: 'lula' },
    { name: 'marina silva' },
    { name: 'paulo rabello' },
    { name: 'rodrigo maia' },
    { name: 'vera lucia' },
    { name: 'candidato a presidente' }
  ];
  this.generateSentimentQueries(terms);
};

exports.realizeSearchForHashtags = async () => {
  const hashs = [
    { name: '#sucesso', sentiment: 'Positivo' },
    { name: '#orgulho', sentiment: 'Positivo' },
    { name: '#falha', sentiment: 'Negativo' },
    { name: '#vergonha', sentiment: 'Negativo' },
    { name: '#nojento', sentiment: 'Negativo' },
    { name: '#ridiculo', sentiment: 'Negativo' },
    { name: '#noticia', sentiment: 'Neutro' },
    { name: '#curiosidade', sentiment: 'Neutro' },
    { name: '#novidade', sentiment: 'Neutro' },
    { name: '#fato', sentiment: 'Neutro' },
    { name: '#oportunidade', sentiment: 'Neutro' },
    { name: '#trabalho', sentiment: 'Neutro' }
  ];
  const writter = defineWritter(CURRENT_WRITTER);
  hashs.forEach((hash) => {
    this.searchQuery(hash.name, hash.sentiment, writter);
  });
};

exports.realizeSearchForPoliticalTerms = async () => {
  const terms = [
    { name: 'autoritario' },
    { name: 'bolsominion' },
    { name: 'centrista' },
    { name: 'centro político' },
    { name: 'comunista' },
    { name: 'congresso' },
    { name: 'conservadorismo' },
    { name: 'discurso esquerda' },
    { name: 'discurso direita' },
    { name: 'eleicoes' },
    { name: 'esquerdismo' },
    { name: 'esquerdista' },
    { name: 'esquerdopata' },
    { name: 'fascista' },
    { name: 'senado' },
    { name: 'governo' },
    { name: 'ideologia esquerda' },
    { name: 'ideologia direita' },
    { name: 'liberal' },
    { name: 'nazista' },
    { name: 'PCdoB' },
    { name: 'PT' },
    { name: 'petista' },
    { name: 'PSDB' },
    { name: 'PSL' },
    { name: 'partido esquerda' },
    { name: 'partido direita' },
    { name: 'PSOL' },
    { name: 'PMDB' },
    { name: 'politica' },
    { name: 'socialista' },
    { name: 'totalitario' },
    { name: 'temer' }
  ];
  this.generateSentimentQueries(terms);
};

exports.streamQuery = async (query, sentimentLabel, writter) => {
  const date = new Date();
  const stream = twit.stream('statuses/filter', { track: query, tweet_mode: 'extended', language: 'pt' });
  const fileName = csvController.determineFileName(query);
  stream.on('tweet', (tweet) => {
    csvController.saveStreamInCSV(tweet, '/Stream/' + fileName, sentimentLabel);
    console.log(date.toLocaleString());
  });
};

exports.searchQuery = async (query, sentimentLabel, writter) => {
  let maxId;
  try {
    const sinceId = await writter.findSinceId(query);
    do {
      const twitterResponse = await twit.get('search/tweets', { q: query, tweet_mode: 'extended', since_id: sinceId, max_id: maxId, count: QUERY_LIMIT, lang: 'pt' });
      const tweets = twitterResponse.data.statuses;
      writter.saveTweets(clearTweets(tweets), query, sentimentLabel);
      maxId = (tweets.length === QUERY_LIMIT) ? tweets[tweets.length - 1].id_str : 0;
    } while (maxId !== 0);
    console.log('end ' + query);
  } catch (err) {
    console.log(err);
  }
};

exports.checkRateLimit = async () => {
  const twitterResponse = await twit.get('application/rate_limit_status');
  console.log(twitterResponse.data.resources.tweets);
};

exports.getTweetById = async (ids, fileName, sentimentLabel) => {
  ids.forEach(async (id) => {
    const twitterResponse = await twit.get('statuses/show', { id, tweet_mode: 'extended', lang: 'pt' });
    console.log(twitterResponse.data.full_text);
    const tweets = twitterResponse.data;
    // csvController.saveTweetsInCSV([tweets], fileName, sentimentLabel);
  });
};

