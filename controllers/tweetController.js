const Twit = require('twit');
const csvController = require('./csvController.js');
const config = require('../config.json');

const QUERY_LIMIT = 100;
const twit = new Twit({
  consumer_key: config.key,
  consumer_secret: config.secret,
  access_token: config.token,
  access_token_secret: config.token_secret,
  timeout_ms: 60 * 1000
});

const generateSentimentQueries = (terms) => {
  terms.forEach((term) => {
    this.searchQuery(term.name + ' :)', 'Positivo', term.sinceIdPos);
    this.searchQuery(term.name + ' :(', 'Negativo', term.sinceIdNeg);
  });
};

exports.realizeSearchForCandidates = async () => {
  const terms = [
    { name: 'alckmin', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'bolsonaro', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'lula', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'aldo rebelo', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'alvaro dias', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'ciro', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'collor', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'flavio rocha', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'guilherme boulos', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'henrique meirelles', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'joao amoedo', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'levy fidelix', sinceIdPos: '', sinceIdNeg: '' },
    { name: "manuela d'avila", sinceIdPos: '', sinceIdNeg: '' },
    { name: 'lula', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'marina silva', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'paulo rabello', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'rodrigo maia', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'vera lucia', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'candidato a presidente', sinceIdPos: '', sinceIdNeg: '' }
  ];
  generateSentimentQueries(terms);
};

exports.realizeSearchForHashtags = async () => {
  const hashs = [
    { name: '#sucesso', sentiment: 'Positivo', since_id: '' },
    { name: '#orgulho', sentiment: 'Positivo', since_id: '' },
    { name: '#falha', sentiment: 'Negativo', since_id: '' },
    { name: '#vergonha', sentiment: 'Negativo', since_id: '' },
    { name: '#nojento', sentiment: 'Negativo', since_id: '' },
    { name: '#ridiculo', sentiment: 'Negativo', since_id: '' },
    { name: '#noticia', sentiment: 'Neutro', since_id: '' },
    { name: '#curiosidade', sentiment: 'Neutro', since_id: '' },
    { name: '#novidade', sentiment: 'Neutro', since_id: '' },
    { name: '#fato', sentiment: 'Neutro', since_id: '' },
    { name: '#oportunidade', sentiment: 'Neutro', since_id: '' },
    { name: '#trabalho', sentiment: 'Neutro', since_id: '' }
  ];
  hashs.forEach((hash) => {
    this.searchQuery(hash.name, hash.sentiment, hash.since_id);
  });
};

exports.realizeSearchForPoliticalTerms = async () => {
  const terms = [
    { name: 'autoritario', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'bolsominion', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'centrista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'centro político', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'comunista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'congresso', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'conservadorismo', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'discurso esquerda', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'discurso direita', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'eleicoes', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'esquerdismo', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'esquerdista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'esquerdopata', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'fascista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'senado', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'governo', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'ideologia esquerda', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'ideologia direita', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'liberal', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'nazista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'PCdoB', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'PT', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'petista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'PSDB', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'PSL', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'partido esquerda', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'partido direita', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'PSOL', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'PMDB', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'politica', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'socialista', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'totalitario', sinceIdPos: '', sinceIdNeg: '' },
    { name: 'temer', sinceIdPos: '', sinceIdNeg: '' }
  ];
  generateSentimentQueries(terms);
};

exports.streamQuery = async (query, sentimentLabel, numberOfTweets = QUERY_LIMIT) => {
  const date = new Date();
  const stream = twit.stream('statuses/filter', { track: query, tweet_mode: 'extended', language: 'pt' });
  const fileName = csvController.determineFileName(query);
  stream.on('tweet', (tweet) => {
    csvController.saveStreamInCSV(tweet, '/Stream/' + fileName, sentimentLabel);
    console.log(date.toLocaleString());
  });
};

exports.searchQuery = async (query, sentimentLabel, sinceId, numberOfTweets = QUERY_LIMIT) => {
  try {
    let tweets;
    let maxId;
    const fileName = csvController.determineFileName(query);
    do {
      const twitterResponse = await twit.get('search/tweets', { q: query, tweet_mode: 'extended', since_id: sinceId, max_id: maxId, count: numberOfTweets, lang: 'pt' });
      tweets = twitterResponse.data.statuses;
      csvController.saveTweetsInCSV(tweets, fileName, sentimentLabel);
      maxId = (tweets.length === numberOfTweets) ? tweets[tweets.length - 1].id_str : 0;
    } while (maxId !== 0);
    console.log('end');
  } catch (err) {
    console.log(err);
  }
};

