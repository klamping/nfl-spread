const { test } = require("@playwright/test");
const { writeFileSync } = require("fs");

const mapTeamName = require("../mapTeamName.js");

const week = 7;
const year = 2024;

test("Get games for week " + week, async ({ page }) => {
  const weekResults = {};
  await page.goto(
    `https://gridirongames.com/nfl-weekly-schedule/?Year=${year}&Week=${week}`,
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
    // const $finalScore = await matchup.locator(
    //   "td:nth-child(5) > div:nth-child(2)"
    // );
    // const finalScore = await $finalScore.textContent();
    // const scoreRegex = /(\d+)\s\-\s(\d+)/g;
    // const scoreMatch = scoreRegex.exec(finalScore);
    // const score = {
    //   away: parseInt(scoreMatch[1]),
    //   home: parseInt(scoreMatch[2]),
    // };
    // stats.score = score;

    // stats.is_favorite_home = stats.favorite === home ? 1 : 0;

    // stats.pointDifferential = score.home - score.away;

    // stats.favoriteCovered = stats.pointDifferential > spread;
    /////

    games.push(stats);
  }

  try {
    const path = `results/matchups/${year}-week-${week}.json`;
    writeFileSync(path, JSON.stringify(games, null, 2), "utf8");
    console.log("Data successfully saved to disk");
  } catch (error) {
    console.log("An error has occurred ", error);
  }
});
