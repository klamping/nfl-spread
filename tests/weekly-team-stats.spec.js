const { test } = require("@playwright/test");
const teams = require("../teams.js");
const { existsSync, writeFileSync } = require("fs");
const season = require('../season-2025.json');

const weekNo = process.env.WEEK_NO || 4;
const historical = process.env.HISTORICAL || false;
let weeks = [season.weeks[weekNo - 1]];

if (historical) {
  const allWeeks = require("./weeks.json");
  weeks = allWeeks.weeks;
}

const oldStats = [
  "points-per-game",
  "average-scoring-margin",
  "opponent-points-per-game",
  "rushing-yards-per-game",
  "passing-yards-per-game",
  "giveaways-per-game",
  "touchdowns-per-game",
  "other-touchdowns-per-game",
  "tackles-per-game",
  "opponent-tackles-per-game",
  "opponent-yards-per-game",
  "opponent-rushing-yards-per-game",
  "opponent-passing-yards-per-game",
  "red-zone-scoring-attempts-per-game",
  // "red-zone-scores-per-game",
  "red-zone-scoring-pct",
  "third-downs-per-game",
  "third-down-conversions-per-game",
  "opponent-third-down-conversion-pct",
  "opponent-third-downs-per-game",
  "opponent-third-down-conversions-per-game",
  "average-time-of-possession-net-of-ot",
  "time-of-possession-pct-net-of-ot",
  "seconds-per-play",
  "qb-sacked-per-game",
  "qb-sacked-pct",
  "sacks-per-game",
  "sack-pct",
  "interceptions-thrown-per-game",
  "fumbles-per-game",
  "takeaways-per-game",
  "turnover-margin-per-game",
  "penalty-yards-per-game",
  "penalties-per-game",
  "penalty-first-downs-per-game",
  "opponent-penalty-yards-per-game",
  "opponent-penalty-first-downs-per-game",
  "penalties-per-play",
];

const stats = [
  'yards-per-game',
  'points-per-game',
  'touchdowns-per-game',
  'extra-points-made-per-game',
  'completion-pct',
  'passing-yards-per-game',
  'passing-first-downs-per-game',
  'passing-touchdown-pct',
  'yards-per-pass-attempt',
  'average-time-of-possession-net-of-ot',
  'rushing-attempts-per-game',  
  'first-downs-per-game',
  'third-down-conversion-pct',
  'fourth-downs-per-game',
  'punt-attempts-per-game',
  'net-punt-yards-per-game',
  'kickoffs-per-game',
  'kicking-points-per-game',
  'turnover-margin-per-game',
  'giveaways-per-game',
  'takeaways-per-game',
  'pass-intercepted-pct',
  'opponent-fumbles-per-game',
  'interceptions-per-game'
]

for (const week of weeks) {
  test.describe(`Get team stats for week of ${week}`, async () => {
    for (const [index, stat] of stats.entries()) {
      test(`Get ${stat} for week ${week}`, async ({ page }) => {
        const weekPath = `./results/weekly-stats/${week}.json`;
        const statPath = `./results/weekly-stats/${week}-${stat}.json`;
        if (existsSync(weekPath) || existsSync(statPath)) {
          // skip the test
          // console.log("Stats already exist, skipping test");
          test.skip();
        }

        console.log(
          `Getting stat ${index + 1} of ${stats.length} for week ${week}`
        );
        await page.goto(
          `https://www.teamrankings.com/nfl/stat/${stat}?date=${week}`,
          {
            waitUntil: "domcontentloaded",
            timeout: 5000,
          }
        );

        const rowLocator = await page.locator("#DataTables_Table_0 tbody tr");
        const teamStats = {};

        let prettyStat = await page.locator("#h1-title").textContent();
        prettyStat = prettyStat.replace("NFL Team ", "");

        for (const teamName of teams) {
          teamStats[teamName] = {};
          const teamRow = await rowLocator.filter({ hasText: teamName });

          // columns: 0 rank, 1 team, 2 this year, 3 last 3, 4 last 1, 5 home, 6 away, 7 last year
          const currentYearStats = await teamRow
            .locator("td")
            .nth(2)
            .textContent();
          const last3Stats = await teamRow.locator("td").nth(3).textContent();
          const last1Stats = await teamRow.locator("td").nth(4).textContent();
          const homeStats = await teamRow.locator("td").nth(5).textContent();
          const awayStats = await teamRow.locator("td").nth(6).textContent();
          teamStats[teamName][`${prettyStat} This Season`] =
            parseFloat(currentYearStats);
          teamStats[teamName][`${prettyStat} Last 3 Games`] =
            parseFloat(last3Stats);
          teamStats[teamName][`${prettyStat} Last 1 Game`] =
            parseFloat(last1Stats);
          teamStats[teamName][`${prettyStat} when Home`] =
            parseFloat(homeStats);
          teamStats[teamName][`${prettyStat} when Away`] =
            parseFloat(awayStats);
        }

        try {
          console.log("Saving weekly team stats to disk");
          writeFileSync(statPath, JSON.stringify(teamStats, null, 2), "utf8");
          console.log("Data successfully saved to disk");
        } catch (error) {
          console.log("An error has occurred ", error);
        }
      });
    }
  });
}
