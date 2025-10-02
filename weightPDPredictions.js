const fs = require('fs');
const { parseArgs } = require('node:util');
const { weeks } = require('./season-2025.json');

const { values } = parseArgs({ 
  options: {
    location: {
      type: 'string',
      short: 'l'
    },
    show: {
      type: 'boolean',
      short: 's'
    }
  }
});
const { location } = values;

const weekNo = process.env.WEEK_NO;
const date = weeks[weekNo - 1];

// Load raw predictions data
const rawPredictionsFile = `./${location}/predictions/raw_predictions_${date}.json`;

// Read and parse both files
const rawPredictions = JSON.parse(fs.readFileSync(rawPredictionsFile, 'utf8'));

// Calculate confidence, weighted confidence, and sort by it
rawPredictions.sort((a, b) => Math.abs(b.Difference) - Math.abs(a.Difference));

// Assign ranks based on weighted confidence
const numPredictions = rawPredictions.length;
const weightedPredictions = rawPredictions.map((item, index) => {
    return {
        ...item,
        WillCoverSpread: item['Will Cover'],
        points: numPredictions - index // Most confident gets the highest rank
    }   
});

// Write the sorted and ranked data to a new file
const outputFilename = `./${location}/predictions/predictions_point_diff_${date}.json`;
fs.writeFileSync(outputFilename, JSON.stringify(weightedPredictions, null, 2), 'utf8');

const parseTeams = (line) => {
  const m = line.match(/^(.+?)\s*\([^)]+\)\s*(?:vs|@)\s*(.+)$/i);
  if (!m) return null;
  const [, team1, team2] = m;
  return { favorite: team1.trim(), underdog: team2.trim() };
};

// Log predictions using console.table
if (values.show) {
    console.log("\nRanked Predictions:\n");
    const prettyResults = weightedPredictions.map((prediction) => {
      // Get who we should pick
      const { favorite, underdog } = parseTeams(prediction.Matchup)
      const puck = prediction['Will Cover'] ? favorite : underdog;

      return {
        'Points': prediction.points,
        Matchup: prediction.Matchup,
        'Puck': puck,
        'Will Cover?': prediction['Will Cover'] ? 'YES' : 'no',
        Prediction: prediction['Prediction'].toFixed(2),
        Difference: prediction['Difference'].toFixed(2)
      }
    });
    console.table(prettyResults);
}
// 
// console.log(`\nRanked predictions have been saved to ${outputFilename}`);
