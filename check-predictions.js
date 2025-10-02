const fs = require('fs');
const path = require('path');
const { parseArgs } = require('node:util');

const { values } = parseArgs({ 
  options: {
    date: {
      type: 'string',
      short: 'd',
    },
    pointDiff: {
      type: 'boolean',
      short: 'p',
    },
    combined: {
      type: 'boolean',
      short: 'c',
    },
    location: {
      type: 'string',
      short: 'l'
    }
  }
});
const { date, pointDiff, combined, location } = values;

// Load JSON game data
const gameData = require('./results/season-matchups/2024.json');

// File paths
let filePath = pointDiff ? 'predictions_point_diff' : 'ranked_predictions'
filePath = combined ? 'combined_points' : filePath;
const predictionsPath = path.join(__dirname, location, 'predictions', `${filePath}_${date}.json`);

const predictions = require(predictionsPath);

function splitGameString(gameString) {
    const regex = /(.+) \((-?\d+\.\d+|-?\d+)\) (vs|@) (.+)/;
    const match = gameString.match(regex);
  
    if (!match) {
      throw new Error('Invalid game string format');
    }
  
    const favorite = match[1].trim();
    const spread = Math.abs(parseFloat(match[2]));
    const isFavoriteHome = match[3] === 'vs';
    const underdog = match[4].trim();
  
    return {
      favorite,
      spread,
      isFavoriteHome,
      underdog
    };
  }


function addResults(predictions, gameResults) {
  return predictions.map(prediction => {
    const { favorite, underdog, isFavoriteHome } = splitGameString(prediction.Matchup);

    const home = isFavoriteHome ? favorite : underdog;
    const away = isFavoriteHome ? underdog : favorite;

    const game = gameResults.find(
      game => 
        game.away.trim().toLowerCase() == away.toLowerCase() &&
        game.home.trim().toLowerCase() == home.toLowerCase()
    );

    if (game) {
      const actualCover = game.favoriteCovered;
      return {
        ...prediction,
        correct: prediction.WillCoverSpread === actualCover
      }
    } else {
        console.error(prediction)
        throw new Error('Game not found')
    }
  });
}

const gameResults = gameData[date];
const updatedPredictions = addResults(predictions, gameResults);
const fileNameAppend = combined ? '_combined' : pointDiff ? '' : '_cover';
const outputFilename = `./${location}/predictions/prediction_results_${date}${fileNameAppend}.json`;
fs.writeFileSync(outputFilename, JSON.stringify(updatedPredictions, null, 2), 'utf8');

console.log(`\nPredictions results have been saved to ${outputFilename}`);
