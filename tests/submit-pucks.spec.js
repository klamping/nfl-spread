const { test } = require('@playwright/test');

test('Submit da pucks', async ({ page }) => {
    await page.goto('https://gridirongames.com/player-login/');

    await page.locator('[data-key="username"]');

    await page.locator('[data-key="user_password"]');

});