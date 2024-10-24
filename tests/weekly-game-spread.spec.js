const { test } = require("@playwright/test");
const { writeFileSync } = require("fs");
const { weeks } = require("./weeks.json");

const mapTeamName = require("../mapTeamName.js");

// const week = 6;
// const path = `results/week-${week}.json`;

const weeksByYear = weeks.reduce((acc, week) => {
  const date = week.split("-");
  let year = date[0];
  const month = date[1];

  if (month === "01") {
    year = year - 1;
  }

  if (!acc[year]) {
    acc[year] = [];
  }

  acc[year].push(week);
  return acc;
}, {});

// const years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023];
// const weeks = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];

for (const [year, weeks] of Object.entries(weeksByYear)) {
  test("Get weekly games for " + year, async ({ page }) => {
    console.log(weeks);
    const weekResults = {};
    for (const [index, week] of weeks.entries()) {
      await page.goto(
        `https://gridirongames.com/nfl-weekly-schedule/?Year=${year}&Week=${
          index + 4
        }`,
          {
              waitUntil: "domcontentloaded"
          }
      );
      const matchupsTable = await page.locator(".nfl-matchups");
      const matchups = await matchupsTable
        .locator('tr[style="height: 55px; border-bottom: 1px solid #aaa;"]')
        .all();

      const games = [];
      for (const matchup of matchups) {
        const awayText = await matchup
          .locator("td:nth-child(2) > div:nth-child(2) > div:nth-child(1)")
          .textContent();
        const away = mapTeamName(awayText);

        const homeText = await matchup
          .locator("td:nth-child(4) > div:nth-child(1) > div:nth-child(1)")
          .textContent();
        const home = mapTeamName(homeText);

        const spreadText = await matchup
          .locator("td:nth-child(3) > div:nth-child(2)")
          .textContent();

        let spreadMatch;
        if (spreadText === "EVEN") {
          spreadMatch = [null, homeText, 0];
        } else {
          const spreadRegex = /(\w+)\s(\-?\d+(?:\.5)?)/g;
          spreadMatch = spreadRegex.exec(spreadText);
        }

        const favorite = mapTeamName(spreadMatch[1]);
        const spread = Math.abs(parseFloat(spreadMatch[2]));

        const stats = {
          away,
          home,
          favorite,
          spread,
        };

        ////
        const $finalScore = await matchup.locator(
          "td:nth-child(5) > div:nth-child(2)"
        );
        const finalScore = await $finalScore.textContent();
        const scoreRegex = /(\d+)\s\-\s(\d+)/g;
        const scoreMatch = scoreRegex.exec(finalScore);
        const score = {
          away: parseInt(scoreMatch[1]),
          home: parseInt(scoreMatch[2]),
        };
        stats.score = score;

        stats.is_favorite_home = stats.favorite === home ? 1 : 0;

        stats.pointDifferential = score.home - score.away;

        stats.favoriteCovered = stats.pointDifferential > spread;
        /////

        games.push(stats);
      }
      weekResults[week] = games;
    }

    console.log("Saving weekly team stats to disk");
    try {
      const path = `results/season-matchups/${year}.json`;
      writeFileSync(path, JSON.stringify(weekResults, null, 2), "utf8");
      console.log("Data successfully saved to disk");
    } catch (error) {
      console.log("An error has occurred ", error);
    }
  });
}
