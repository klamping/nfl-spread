const fs = require('fs');
const path = require('path');
const { parseArgs } = require('node:util');

const { values } = parseArgs({ 
  options: {
    date: {
      type: 'string',
      short: 'd',
    },
    location: {
      type: 'string',
      short: 'l'
    }
  }
});
const { date, location } = values;

const resultsFilename = `./${location}/predictions/prediction_results_${date}.json`;

const weekResults = require(resultsFilename);