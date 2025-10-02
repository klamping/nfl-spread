const { readFileSync, writeFileSync } = require("fs");
const { parseArgs } = require('node:util');
const path = require("path");
const { weeks } = require('./season-2025.json');

// Load the CSV file with feature list
const csvFilePath = path.resolve(__dirname, "tsseven", "Top_95__Cumulative_Contribution_Features.csv");
const csvData = readFileSync(csvFilePath, "utf8");

// Parse CSV data into a list of valid features
const validFeatures = csvData.split("\n").slice(1).map(row => row.split(",")[0].trim());

// Function to check if a stat is valid
const isValidStat = statName => true//validFeatures.includes(statName);

const { values } = parseArgs({ 
  options: {
    pointDiff: {
      type: 'boolean',
      short: 'p'
    }
  }
});


const weekNo = process.env.WEEK_NO;
const date = weeks[weekNo - 1];
const useSpecificWeek = !!(weekNo);
const usePointDiff = values.pointDiff;
const splitSpread = false;
const splitHome = !useSpecificWeek && false;

let stats, seasons, outputPath;

const fileAppend = usePointDiff ? '-pointDiff' : '';

if (useSpecificWeek) {
  stats = require(`./${date}-team-stats.json`);
  const weekMatchups = require(`./results/matchups/2025-week-${weekNo}.json`);
  seasons = {
    [date]: weekMatchups
  };
  outputPath = `./model-data/${date}-lines${fileAppend}`;
} else {
  stats = require("./all-weekly-data-3.json");
  seasons = require("./seasons.json");
  outputPath = `./model-data/seasons-no-2025${fileAppend}`;
}

// This file will combine the weekly team stats, and the weekly game results, to create a CSV file that can be used to train a machine learning model.
let output = "";

const headings = [];
if (useSpecificWeek) {
  headings.push("Matchup");
}
if (!usePointDiff) {
  headings.push("Spread");
}
// headings.push("Week");
// headings.push("Day");
// headings.push("Time");

if (!splitHome) {
  headings.push("Is Favorite Home Team");
}
headings.push("Win Percent Diff");
const firstWeek = Object.keys(stats)[0];
const firstTeam = Object.keys(stats[firstWeek])[0];
const firstTeamStats = Object.keys(stats[firstWeek][firstTeam]);
const numStats = firstTeamStats.length;
for (const stat of firstTeamStats) {
  if (stat.includes(' when Away') || 
    stat.includes('Red Zone') || 
    stat.includes('Passing Touchdown Percentage') ||
    !isValidStat(stat)
  ) {
    continue;
  } else if (stat.includes(' when Home')) {
    // split the state out from 'when' value
    const normalizedStat = stat.split(' when')[0];
    headings.push(`${normalizedStat} @ vs. season average`);
  } else if (stat.includes(' Last ')) {
    headings.push(`${stat} vs. season average`);
  } else {
    headings.push(`${stat}`);
  }
}

if (!useSpecificWeek) {
  headings.push('Target');
}
output += headings.join(",") + "\n";

function getMainStatName(stat) {
  return stat
      .split(/when|Last/)[0] // Split on "when" or "Last" and take the first part
      .trim();               // Trim any trailing spaces
}

function calcStatDiff(statName, stat, stats) {
  // get the season stat value
  const mainStatName = getMainStatName(statName);
  const mainStat = stats[mainStatName + ' This Season'];
  const diffStat = mainStat - stat; 
  return diffStat.toFixed(2);
}

const spreadResults = {};
const outputSplit = {
  homeFave: output,
  awayFave: output
}

for (const week of Object.keys(seasons)) {
  // skip historical weeks from 2025
  if (!useSpecificWeek && week.includes('2025') && !week.includes('2025-01')) {
    console.log(`Skipping week ${week} because it's in the 2025 season`)
    continue;
  }

  const games = seasons[week];
  for (const game of games) {
    if (!stats[week]) {
      console.log("No stats for week", week);
      continue;
    }
    // don't worry about zero spreads, as they're always home, and can throw off stats
    if (game.spread === 0) {
      continue;
    }

    if (!useSpecificWeek && !usePointDiff) {
      const faveWonBy = game.favorite === game.home ? 
        game.score.home - game.score.away :
        game.score.away - game.score.home;
      if (faveWonBy === game.spread) {
        console.log('spread matches point diff');
        continue;
      }
    }

    const isFaveHome = useSpecificWeek ? 
      game.favorite === game.home :
      !!game.is_favorite_home;

    const favoriteStats = stats[week][game.favorite];
    const underdog = isFaveHome ? game.away : game.home;
    const underdogStats = stats[week][underdog];

    const data = [];
    let hasNullStats = false;

    if (useSpecificWeek) {
      data.push(`${game.favorite} (-${game.spread}) ${isFaveHome ? 'vs' : '@'} ${underdog}`);
    }

    if (!usePointDiff) {
      data.push(game.spread);
    }
    if (!splitHome) {
      data.push(isFaveHome ? 1 : 0);
    }

    const faveRecord = isFaveHome ? game.smoothedHomeRecord : game.smoothedAwayRecord;
    const underdogRecord = isFaveHome ? game.smoothedAwayRecord : game.smoothedHomeRecord;

    data.push(faveRecord - underdogRecord);

    if (Object.keys(favoriteStats).length !== numStats) {
      const theStats = Object.keys(favoriteStats);
      // Compare the stats and the main stats
      // Find elements in list1 but not in list2
      const difference_1 = theStats.filter(item => !firstTeamStats.includes(item) && item.includes('Last 1 game'));

      // Find elements in list2 but not in list1
      const difference_2 = firstTeamStats.filter(item => !theStats.includes(item) && item.includes('Last 1 game'));

      if (difference_1.length === 0 && difference_2.length === 0) {
        continue;
      }

      // Print the differences
      console.log("Elements in list1 but not in firstTeamStats:", difference_1);
      console.log("Elements in firstTeamStats but not in list1:", difference_2);

      throw new Error(`Missing stats for week ${week} and team ${game.favorite}.`);
    }
    for (const [statName, stat] of Object.entries(favoriteStats)) {
      if (
        (!isFaveHome && statName.includes(' when Home')) ||
        (isFaveHome && statName.includes(' when Away')) ||
        statName.includes('Red Zone') ||
        statName.includes('Passing Touchdown Percentage') ||
        !isValidStat(stat)
      ) {
        // ignore stat
        continue;
      } else {
        let underdogStatName = statName;
        if (statName.includes(' when ')) {
          const mainStatName = getMainStatName(statName);
          underdogStatName = `${mainStatName} when ${isFaveHome ? 'Away' : 'Home'}`; 
        }
        const underdogStat = underdogStats[underdogStatName];
        if (stat === null || underdogStat === null) {
          if (!hasNullStats) {
            console.log('no stat found for', statName, stat, underdogStat)
          }
          hasNullStats = true;
          data.push(0);
        } else if (statName.includes(' This Season')) {
          const statDiff = stat - underdogStat;
          data.push(statDiff);
        } else {
          const faveDiffStat = calcStatDiff(statName, stat, favoriteStats);
          const underdogDiffStat = calcStatDiff(underdogStatName, underdogStats[underdogStatName], underdogStats)
          if (!faveDiffStat || !underdogDiffStat) {
            throw new Error('Data error for', statName, underdogStatName)
          }
          data.push(faveDiffStat - underdogDiffStat);
        }
      }
    }

    if (!useSpecificWeek) {
      if (usePointDiff) {
        data.push(game.faveWonBy);
      } else {
        data.push(game.favoriteCovered ? 1 : 0);
      }
    }

    if (!hasNullStats) {
      if (useSpecificWeek || !splitSpread) {
        if (splitHome) {
          const splitProp = isFaveHome ? 'homeFave' : 'awayFave';
          outputSplit[splitProp] += data.join(",") + "\n";
        } else {
          output += data.join(",") + "\n";
        }
      } else {
        if (!Object.hasOwn(spreadResults, game.spread)) {
          spreadResults[game.spread.toString()] = '';
        }
        spreadResults[game.spread.toString()] += data.join(",") + "\n";
      }
    } else {
      console.log('skipping week because has null stats')
    }
  }
}

const spreads = {
  lower: [0.5, 1, 1.5, 2, 2.5, 3, 3.5],
  middle: [4, 4.5, 5, 5.5],
  middleUp: [6, 6.5, 7, 7.5, 8, 8.5],
  upper: [9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5],
  one: [0.5, 1, 1.5, 2, 2.5],
  oneTwoThree: [1.5, 2, 2.5, 3, 3.5],
  twoThreeFour: [2.5, 3, 3.5, 4, 4.5],
  four: [4, 4.5],
  fourFiveSix: [4, 4.5, 5, 5.5, 6, 6.5],
  sevenEightNine: [7, 7.5, 8, 8.5, 9, 9.5],
  aboveNine: [10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5],
  two: [0.5, 1, 1.5, 2, 2.5],
  three: [3, 3.5],
  four: [4, 4.5],
  fiveSix: [5, 5.5, 6, 6.5], 
  seven: [7, 7.5],
  eight: [7.5, 8, 8.5, 9],
  aboveEight: [9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5]
}

try {
  // save the three files
  if (useSpecificWeek || !splitSpread) {
    if (splitHome) {
      for (const place of ['homeFave', 'awayFave']) {
        const filePath = `${outputPath}-${place}.csv`;
        writeFileSync(filePath, outputSplit[place], "utf8");
        console.log("Data successfully saved to disk", filePath);
      }
    } else {
      writeFileSync(`${outputPath}.csv`, output, "utf8");
      console.log("Data successfully saved to disk", outputPath);
    }
  } else {
    for (const [label, spreadValues] of Object.entries(spreads)) {
      let results = output;
      for (const spread of spreadValues) {
        results += spreadResults[spread];
      }
      const fileName = `${outputPath}-${label}.csv`;
      writeFileSync(fileName, results, "utf8");
      console.log("Data successfully saved to disk", fileName);
    }
  }
} catch (error) {
  console.log("An error has occurred ", error);
}