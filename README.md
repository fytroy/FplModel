# FPL Autopilot

Predicts Fantasy Premier League points, plans your squad several weeks ahead, makes your free transfers, sets your captain and bench before each deadline, and tells you on your phone what it did.

Built on [fpl-model](https://github.com/savirmediratta123/fpl-model) by Savir Mediratta, which has no licence. Keep this repo private unless the author gives permission to publish it.

## Risks to accept before switching autopilot on

- FPL has no official transfer API. Autopilot uses the same private calls as the FPL website, and they can change without notice.
- A third-party reading of FPL's terms says they bar automated access. Read the [official terms](https://fantasy.premierleague.com/help/terms) yourself: a breach could cost you your account.
- Transfers can't be undone. Predictions are estimates, so some moves will turn out badly.
- If FPL blocks the login or a request, autopilot stops and alerts you. It does not try to get around a block.

## What autopilot will and won't do

It runs once, about 1 to 3.5 hours before each deadline, after press-conference news.

| It will | It won't |
| --- | --- |
| Make the best moves, up to 2 free transfers a week | Take a points hit |
| Set the starting XI, bench order, captain and vice-captain | Play a chip |
| Use your real selling prices, bank and free transfers | Touch your team if you've already made a transfer that week |
| Read your team back to confirm the changes landed | Act within 15 minutes of the deadline |
| Notify you of what it did, or why it stood down | Buy a player who isn't fully available |

If the model prefers a hit, a chip or more than 2 transfers, autopilot makes the best plan within those limits, and the notification tells you what the model wanted so you can decide.

## Setup

**1. Create your private repo.** Make an empty private repo on GitHub, then run:

```bash
git remote add mine https://github.com/<your-username>/<your-repo>.git
```

```bash
git push -u mine autopilot:master
```

In the repo's Settings → General, check that the default branch is `master`.

**2. Find your team ID.** Open the Points page on the FPL site. The number after `/entry/` in the URL is your ID.

**3. Set up notifications.** Install the ntfy app and subscribe to a long, random topic name of your own. Anyone who knows the name can read it.

**4. Add settings.** In the repo, go to Settings → Secrets and variables → Actions, and add these repository secrets:

| Secret | Value |
| --- | --- |
| `FPL_TEAM_ID` | Your team ID |
| `NTFY_TOPIC` | Your ntfy topic |

**5. Test notifications.** In the Actions tab, open "FPL deadline alerts", choose Run workflow, then pick `check`. A test message should reach your phone. Then run `dry-run` and read the advice in the log.

Stop here if you only want notifications.

**6. Add your login token, for autopilot.** The token gives access to your FPL account, so treat it like a password and never share it or commit it.

1. Log in at fantasy.premierleague.com in your browser.
2. Open DevTools, then Application → Local Storage → `https://fantasy.premierleague.com`.
3. Copy the value of the key that starts with `oidc.user:`.
4. Add it as the secret `FPL_REFRESH_TOKEN`.

**7. Let the system save its new token.** FPL replaces the token on every use, so the workflow has to update the secret itself. Create a [fine-grained personal access token](https://github.com/settings/personal-access-tokens/new) limited to this repo, with the Secrets permission set to read and write. Add it as the secret `SECRET_WRITE_TOKEN`.

**8. Rehearse.** Add a repository variable (not a secret) `FPL_AUTOPILOT` = `on`. Then run the workflow in `dry-run` mode. It logs in, reads your team and shows what it would do, without changing anything. The first scheduled run is your first real one, so do this rehearsal before the next deadline.

To turn autopilot off at any time, set `FPL_AUTOPILOT` to `off`. Notifications carry on.

For full optimizer automation, add these repository variables:

| Variable | Value |
| --- | --- |
| `FPL_AUTOPILOT` | `on` |
| `FPL_AUTOPILOT_POLICY` | `all_planned` |
| `FPL_MESSAGE_FREQUENCY` | `updated_only` |

This applies the planned transfers, hits and chips in the late deadline window, sets the
captain and bench, reads the team back to verify the update, and sends an ntfy message
only after an actual update or when the run is blocked or fails. Keep the FPL login and
secret-write token in GitHub Secrets; never put them in repository variables.

## If it stops working

- **"FPL login failed":** the token has expired or been replaced. Repeat step 6.
- **"Couldn't save the new login token":** check `SECRET_WRITE_TOKEN`, then repeat step 6.
- **"autopilot stood down":** a safety check said no, and your team wasn't changed. The message gives the reason.
- **"autopilot FAILED":** open FPL and check your team, because some changes may have gone through.

## Run it on your own computer

Copy `.env.example` to `.env`, fill it in, then:

```bash
pip install -r requirements.txt
```

```bash
python pull_history.py
```

```bash
python recommend.py
```

```bash
python alert.py run --stage late --dry-run
```

`CLAUDE.md` lists every command and how the system works.
"# FplModel" 
