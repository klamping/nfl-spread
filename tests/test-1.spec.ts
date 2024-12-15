import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  // login 
  await page.goto('https://gridirongames.com/player-login/');
  await page.getByLabel('Username or E-mail*').click();
  await page.getByLabel('Username or E-mail*').fill('klamping');
  await page.getByLabel('Password*').click();
  await page.getByLabel('Password*').fill('SpursGr1dF4n09');
  await page.getByRole('button', { name: 'Log In' }).click();
  await page.getByRole('button', { name: 'View Pool' }).click();
  // set pucks
  await page.getByRole('cell', { name: 'DET DET (11-1-0) (-3.5)' }).locator('span').nth(3).click();
  await page.locator('select[name="P1"]').selectOption('');
  await page.locator('select[name="P1"]').selectOption('1');

  // final
  await page.getByPlaceholder('Enter total combined score').click();
  await page.getByPlaceholder('Enter total combined score').fill('22');
  await page.getByRole('button', { name: 'Submit Week 14 Picks' }).click();
  await page.getByText('× Week 14 picks successfully').click();
  await expect(page.getByRole('alert')).toContainText('× Week 14 picks successfully submitted.');
});
