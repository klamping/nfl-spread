/// OLD DO NOT USE
/// USE 'createCSV' instead

const week = "2024-11-20";
const weekNo = "12";

const stats = require(`./${week}-team-stats.json`);
const matchups = require(`./results/matchups/2024-week-${weekNo}.json`);
const { writeFileSync } = require("fs");

const path = `results/${week}-lines.csv`;

// This file will combine the weekly team stats, and the weekly game results, to create a CSV file that can be used to train a machine learning model.
let output = "";


const headings = [];
headings.push("Spread");
headings.push("Week");
headings.push("Day");
headings.push("Time");
headings.push("Is Favorite Home Team");
headings.push("Favorite Win Percent");
const firstTeam = Object.keys(stats[week])[0];
for (const stat of Object.keys(stats[week][firstTeam])) {
  if (stat.includes('Last 1 Game') || stat.includes(' when Away')) {
    continue;
  } else if (stat.includes(' when Home')) {
    // split the state out from 'when' value
    const normalizedStat = stat.split(' when')[0];
    headings.push(`Favorite: ${normalizedStat} @`);
  } else {
    headings.push(`Favorite: ${stat}`);
  }
}
headings.push("Underdog Win Percent");
for (const stat of Object.keys(stats[week][firstTeam])) {
  if (stat.includes('Last 1 Game') || stat.includes(' when Away')) {
    continue;
  } else if (stat.includes(' when Home')) {
    // split the state out from 'when' value
    const normalizedStat = stat.split(' when')[0];
    headings.push(`Underdog: ${normalizedStat} @`);
  } else {
    headings.push(`Underdog: ${stat}`);
  }
}
headings.push("Matchup");
output += headings.join(",") + "\n";

for (const matchup of matchups) {
  const isFavoriteHome = matchup.favorite === matchup.home;
  const underdog = isFavoriteHome ? matchup.away : matchup.home;
  const favoriteStats = stats[week][matchup.favorite];
  const underdogStats = stats[week][underdog];

  const data = [];

  data.push(matchup.spread);
  data.push(matchup.nflWeek);
  data.push(matchup.dayOfWeek);
  data.push(matchup.time);
  data.push(isFavoriteHome ? 1 : 0);

  data.push(isFavoriteHome ? matchup.homeRecord : matchup.awayRecord);

  for (const [statName, stat] of Object.entries(favoriteStats)) {
    if (
      statName.includes('Last 1 Game') ||
      (!isFavoriteHome && statName.includes(' when Home')) ||
      (isFavoriteHome && statName.includes(' when Away'))
    ) {
      // ignore stat
      continue;
    } else {
      if (statName.includes('Red Zone') && stat === null) {
        data.push(0);
      } else if (statName.includes(' when ') && stat === null) {
        // if we don't have stats for the location, just use season stats
        const mainStat = statName.split(' when ')[0];
        const stat = favoriteStats[mainStat + ' This Season'];
        data.push(stat);
      } else {
        data.push(stat);
      }
    }
  }

  data.push(isFavoriteHome ? matchup.awayRecord : matchup.homeRecord);
  for (const [statName, stat] of Object.entries(underdogStats)) {
    if (
      statName.includes('Last 1 Game') ||
      (!isFavoriteHome && statName.includes(' when Away')) ||
      (isFavoriteHome && statName.includes(' when Home'))
    ) {
      continue;
    } else {
      if (statName.includes('Red Zone') && stat === null) {
        data.push(0);
      } else if (statName.includes(' when ') && stat === null) {
        // if we don't have stats for the location, just use season stats
        const mainStat = statName.split(' when ')[0];
        const stat = underdogStats[mainStat + ' This Season'];
        data.push(stat);
      } else {
        data.push(stat);
      }
    }
  }
  // data.push(matchup.favoriteCovered ? 1 : 0);
  data.push(`${matchup.favorite} (-${matchup.spread}) ${isFavoriteHome ? 'vs' : '@'} ${underdog}`);
  output += data.join(",") + "\n";
}

try {
  writeFileSync(path, output, "utf8");
  console.log("Data successfully saved to disk", path);
} catch (error) {
  console.log("An error has occurred ", error);
}
