const { writeFileSync } = require("fs");

const week = "2024-10-16";
const year = 2024;
const weekNo = 7;
const stats = require(`./${week}team-stats.json`);
const matchups = require(`./results/matchups/${year}-week-${weekNo}.json`);
const path = `results/${week}-matchups.csv`;

const skippedStat = "Red Zone Scoring Percentage (TD only) Last 1 Game";
// This file will combine the weekly team stats, and the weekly game results, to create a CSV file that can be used to train a machine learning model.
let output = "";

const headings = [];
const firstWeek = Object.keys(stats)[0];
const firstTeam = Object.keys(stats[firstWeek])[0];
headings.push("Spread");

for (const stat of Object.keys(stats[firstWeek][firstTeam])) {
  if (stat === skippedStat) {
    continue;
  }
  headings.push(`Home: ${stat}`);
}
for (const stat of Object.keys(stats[firstWeek][firstTeam])) {
  if (stat === skippedStat) {
    continue;
  }
  headings.push(`Away: ${stat}`);
}
headings.push("Matchup");
output += headings.join(",") + "\n";

for (const game of matchups) {
  if (!stats[week]) {
    console.log("No stats for week", week);
    continue;
  }
  const home = stats[week][game.home];
  const away = stats[week][game.away];

  const data = [];

  const isHomeFavorite = game.favorite === game.home;
  const spread = isHomeFavorite ? game.spread : -game.spread;
  data.push(spread);

  for (const [statName, stat] of Object.entries(home)) {
    if (statName === skippedStat) {
      continue;
    }
    data.push(stat);
  }
  for (const [statName, stat] of Object.entries(away)) {
    if (statName === skippedStat) {
      continue;
    }
    data.push(stat);
  }

  const underdog = isHomeFavorite ? game.away : game.home;
  data.push(`${game.favorite} (-${game.spread}) ${isHomeFavorite ? 'vs' : '@'} ${underdog}`);

  // data.push(game.pointDifferential);

  output += data.join(",") + "\n";
}

try {
  writeFileSync(path, output, "utf8");
  console.log("Data successfully saved to disk");
} catch (error) {
  console.log("An error has occurred ", error);
}
