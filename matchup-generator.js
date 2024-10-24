const week = "2024-10-09";
const weekNo = "6";

const stats = require(`./results/weekly-stats/${week}.json`);
const matchups = require(`./results/week-${weekNo}.json`);
const { writeFileSync } = require("fs");

const path = `results/${week}-lines.csv`;

// This file will combine the weekly team stats, and the weekly game results, to create a CSV file that can be used to train a machine learning model.
let output = "";

const headings = [];
headings.push("Spread");
headings.push("Is Favorite Home Team");
const firstTeam = Object.keys(stats)[0];
for (const stat of Object.keys(stats[firstTeam])) {
  headings.push(`Favorite: ${stat}`);
}
for (const stat of Object.keys(stats[firstTeam])) {
  headings.push(`Underdog: ${stat}`);
}
headings.push("Favorite Covered Spread");
headings.push("Matchup");
output += headings.join(",") + "\n";

for (const matchup of matchups) {
  const isFavoriteHome = matchup.favorite === matchup.home;
  const underdog = isFavoriteHome ? matchup.away : matchup.home;
  const favoriteStats = stats[matchup.favorite];
  const underdogStats = stats[underdog];

  const data = [];

  data.push(matchup.spread);
  data.push(isFavoriteHome);
  for (const stat of Object.values(favoriteStats)) {
    data.push(stat);
  }
  for (const stat of Object.values(underdogStats)) {
    data.push(stat);
  }

  data.push(matchup.favoriteCovered ? 1 : 0);
  data.push(`${matchup.favorite} (-${matchup.spread}) vs ${underdog}`);
  output += data.join(",") + "\n";
}

try {
  writeFileSync(path, output, "utf8");
  console.log("Data successfully saved to disk", path);
} catch (error) {
  console.log("An error has occurred ", error);
}
