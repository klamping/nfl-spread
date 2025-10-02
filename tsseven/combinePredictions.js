// Import the required modules
const fs = require('fs');
const { parseArgs } = require('node:util');

const { values } = parseArgs({ 
  options: {
    date: {
      type: 'string',
      short: 'd',
    }
  }
});
const { date } = values;

// Load the JSON files
const coverFile = __dirname + `/predictions/raw_predictions_${date}_cover.json`;
const predictionsFile = __dirname + `/predictions/raw_predictions_${date}.json`;

// Function to load JSON data from a file
const loadJson = (filePath) => {
    try {
        const data = fs.readFileSync(filePath, 'utf-8');
        return JSON.parse(data);
    } catch (error) {
        console.error(`Error reading or parsing file: ${filePath}`, error);
        return null;
    }
};

// Function to combine and compare predictions
const combineAndComparePredictions = (coverData, predictionsData) => {
    return coverData.map((coverItem) => {
        const matchPrediction = predictionsData.find(
            (predItem) => predItem.Matchup === coverItem.Matchup
        );

        const bothAgree = (coverItem["Will Cover"] === matchPrediction["Will Cover"]);

        return {
            Matchup: coverItem.Matchup,
            Spread: coverItem.Spread,
            CoverPrediction: !!coverItem.Prediction,
            PointDiffPrediction: matchPrediction['Will Cover'],
            PointDiff: Math.abs(matchPrediction.Difference),
            Agree: bothAgree,
            Covers: matchPrediction['Will Cover']
        };
    }).sort((a, b) => {
        // Sort by 'Agree' property (false comes before true)
        // if (a.Agree !== b.Agree) {
          return b.Agree - a.Agree;
        // }
        // If 'Agree' is the same, sort by 'PointDiff' property
        // return b.PointDiff - a.PointDiff;
    });
};

// Function to display data as a table
const displayTable = (combinedData) => {
    console.table(combinedData, ["Matchup", "CoverPrediction", "PointDiffPrediction", "Agree", "Covers", "PointDiff", "points"]);
};

// Main execution
const coverData = loadJson(coverFile);
const predictionsData = loadJson(predictionsFile);

if (coverData && predictionsData) {
    const combinedData = combineAndComparePredictions(coverData, predictionsData);

    const numPredictions = combinedData.length;
    const weightedPredictions = combinedData.map((item, index) => {
        return {
            ...item,
            WillCoverSpread: item['Covers'],
            points: numPredictions - index // Most confident gets the highest rank
        }   
    });

    // Save combined data to a new JSON file
    const outputFilePath = __dirname + `/predictions/combined_points_${date}.json`;
    fs.writeFileSync(outputFilePath, JSON.stringify(weightedPredictions, null, 2), 'utf-8');

    console.log(`Combined predictions saved to ${outputFilePath}`);

    // Display the table
    // displayTable(weightedPredictions);
} else {
    console.error('Failed to load input files.');
}
