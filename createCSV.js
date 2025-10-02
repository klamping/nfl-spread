const { writeFileSync } = require("fs");
const { parseArgs } = require('node:util');
const { weeks } = require('./season-2025.json');

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
if (!usePointDiff) {
  headings.push("Spread");
}
// headings.push("Week");
// headings.push("Day");
// headings.push("Time");
headings.push("Is Favorite Home Team");
headings.push("Favorite Win Percent");
const firstWeek = Object.keys(stats)[0];
const firstTeam = Object.keys(stats[firstWeek])[0];
const firstTeamStats = Object.keys(stats[firstWeek][firstTeam]);
const numStats = firstTeamStats.length;
for (const stat of firstTeamStats) {
  if (stat.includes(' when Away')) {
    continue;
  } else if (stat.includes(' when Home')) {
    // split the state out from 'when' value
    const normalizedStat = stat.split(' when')[0];
    headings.push(`Favorite: ${normalizedStat} @ vs. season average`);
  } else if (stat.includes(' Last ')) {
    headings.push(`Favorite: ${stat} vs. season average`);
  } else {
    headings.push(`Favorite: ${stat}`);
  }
}
headings.push("Underdog Win Percent");
for (const stat of firstTeamStats) {
  if (stat.includes(' when Away')) {
    continue;
  } else if (stat.includes(' when Home')) {
    // split the state out from 'when' value
    const normalizedStat = stat.split(' when')[0];
    headings.push(`Underdog: ${normalizedStat} @ vs. season average`);
  } else if (stat.includes(' Last ')) {
    headings.push(`Underdog: ${stat} vs. season average`);
  } else {
    headings.push(`Underdog: ${stat}`);
  };
}

if (useSpecificWeek) {
  headings.push("Matchup");
} else if (usePointDiff) {
  // headings.push("Game Result: Favorite Points");
  // headings.push("Game Result: Underdog Points");
  headings.push("Favorite Won By");
} else {
  headings.push("Favorite Covered Spread");
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

    if (!usePointDiff) {
      data.push(game.spread);
    }
    // data.push(game.nflWeek);
    // data.push(game.dayOfWeek);
    // data.push(game.time);
    data.push(isFaveHome ? 1 : 0);

    data.push(isFaveHome ? game.smoothedHomeRecord : game.smoothedAwayRecord);

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
        (isFaveHome && statName.includes(' when Away'))
      ) {
        // ignore stat
        continue;
      } else {
        if (statName.includes(' This Season')) {
          data.push(stat);
        } else if (stat === null) {
          data.push(0);
        } else {
          const diffStat = calcStatDiff(statName, stat, favoriteStats);
          data.push(diffStat);
        }
      }
    }

    data.push(isFaveHome ? game.smoothedAwayRecord : game.smoothedHomeRecord);

    if (Object.keys(underdogStats).length !== numStats) {
      throw new Error(`Missing stats for week ${week} and team ${game.underdog}. ${numStats}, ${Object.keys(underdogStats).length}`);
    }
    for (const [statName, stat] of Object.entries(underdogStats)) {
      if (
        (!isFaveHome && statName.includes(' when Away')) ||
        (isFaveHome && statName.includes(' when Home'))
      ) {
        continue;
      } else {
        if (statName.includes(' This Season')) {
          data.push(stat);
        } else if (stat === null) {
          data.push(0);
        } else {
          const diffStat = calcStatDiff(statName, stat, underdogStats);
          data.push(diffStat);
        }
      }
    }

    // data.push(game.favoriteScore);
    // data.push(game.underdogScore);
    if (useSpecificWeek) {
      data.push(`${game.favorite} (-${game.spread}) ${isFaveHome ? 'vs' : '@'} ${underdog}`);
    } else if (usePointDiff) {
      data.push(game.faveWonBy);
    } else {
      data.push(game.favoriteCovered ? 1 : 0);
    }

    if (useSpecificWeek || !splitSpread) {
      output += data.join(",") + "\n";
    } else {
      if (!Object.hasOwn(spreadResults, game.spread)) {
        spreadResults[game.spread.toString()] = '';
      }
      spreadResults[game.spread.toString()] += data.join(",") + "\n";
    }
  }
}

const spreads = {
  // oneTwoThree: [1.5, 2, 2.5, 3, 3.5]
  // a: [0.5, 1, 1.5, 2, 2.5],
  // twoThreeFour: [2.5, 3, 3.5, 4, 4.5],
  // four: [4, 4.5],
  // fourFiveSix: [4, 4.5, 5, 5.5, 6, 6.5],
  // sevenEightNine: [7, 7.5, 8, 8.5, 9, 9.5],
  // aboveNine: [10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5]
  // twoFive: [2, 2.5],
  // threeFive: [3, 3.5],
  // four: [4, 4.5],
  // fiveSix: [5, 5.5, 6, 6.5], 
  // seven: [7, 7.5],
  eight: [7.5, 8, 8.5, 9],
  // aboveEight: [9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5]
}

try {
  // save the three files
  if (useSpecificWeek || !splitSpread) {
    writeFileSync(`${outputPath}.csv`, output, "utf8");
    console.log("Data successfully saved to disk", outputPath);
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